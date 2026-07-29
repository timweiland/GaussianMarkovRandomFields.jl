using GaussianMarkovRandomFields
using Distributions
using SparseArrays
using LinearAlgebra
using LinearSolve
using Statistics

using DifferentiationInterface
using FiniteDiff, ForwardDiff
using ReTest: @testset, @test, @test_throws

# GMRF types that carry Dual numbers deliberately keep a *primal* factorization —
# CHOLMOD cannot hold Duals — so anything that reads marginal variances straight
# out of that factorization silently drops its partials and reports a zero
# gradient. These tests pin `var`/`std` to dense references that share no code
# with the selected-inversion path.
#
# The precision below is a 2D grid Laplacian: its Cholesky factor has fill-in, so
# entries of Q⁻¹ outside the selected-inverse pattern genuinely matter. A
# tridiagonal AR(1) precision has no fill-in and cannot detect this class of bug.
function _fd_var_grid_laplacian(m)
    I_, J_ = Int[], Int[]
    for j in 1:m, i in 1:m
        idx = (j - 1) * m + i
        i < m && (push!(I_, idx); push!(J_, idx + 1))
        j < m && (push!(I_, idx); push!(J_, idx + m))
    end
    W = sparse([I_; J_], [J_; I_], 1.0, m * m, m * m)
    return spdiagm(0 => vec(sum(W, dims = 2))) - W
end

@testset "ForwardDiff var/std" begin
    L = _fd_var_grid_laplacian(8)
    N = size(L, 1)
    Qof(θ) = sparse(exp(θ[1]) * (L + exp(θ[2]) * I))
    θ0 = [log(2.0), 0.3]

    # Dense marginal variances — no GMRF machinery, no selected inversion.
    dense_var(θ) = diag(inv(Matrix(Qof(θ))))

    # Guard the premise: without fill-in this suite cannot see the bug it targets.
    @testset "test precision has Cholesky fill-in" begin
        Q = Qof(θ0)
        @test nnz(sparse(cholesky(Symmetric(Q)).L)) > nnz(tril(Q))
    end

    @testset "var returns a dense Vector" begin
        # `selinv_diag` yields a structurally dense `SparseVector`, which
        # `ForwardDiff.jacobian` cannot extract partials from.
        Q = Qof(θ0)
        @test var(GMRF(zeros(N), Q, LinearSolve.CHOLMODFactorization())) isa Vector
        @test var(GMRF(zeros(N), Q)) isa Vector
        @test var(ChordalGMRF(zeros(N), Q)) isa Vector
        @test var(WorkspaceGMRF(zeros(N), Q)) isa Vector
    end

    @testset "gradient of sum(var)" begin
        ref = ForwardDiff.gradient(θ -> sum(dense_var(θ)), θ0)
        @test ref ≉ zeros(2)  # the reference must be non-trivial

        # `GMRF(μ, Q)` resolves to CHOLMOD for sparse SPD precisions, so the
        # default constructor must be differentiable too — not just the
        # explicitly-configured one.
        gmrf_sum(θ) = sum(var(GMRF(zeros(N), Qof(θ), LinearSolve.CHOLMODFactorization())))
        default_sum(θ) = sum(var(GMRF(zeros(N), Qof(θ))))
        chordal_sum(θ) = sum(var(ChordalGMRF(zeros(N), Qof(θ))))
        workspace_sum(θ) = sum(var(WorkspaceGMRF(zeros(N), Qof(θ))))

        @test ForwardDiff.gradient(gmrf_sum, θ0) ≈ ref rtol = 1.0e-8
        @test ForwardDiff.gradient(default_sum, θ0) ≈ ref rtol = 1.0e-8
        @test ForwardDiff.gradient(chordal_sum, θ0) ≈ ref rtol = 1.0e-8
        @test ForwardDiff.gradient(workspace_sum, θ0) ≈ ref rtol = 1.0e-8

        # Independent check that the dense reference itself is right.
        @test FiniteDiff.finite_difference_gradient(gmrf_sum, θ0) ≈ ref rtol = 1.0e-5
    end

    @testset "full Jacobian of var" begin
        # A sum can hide per-element sign errors; pin every marginal variance.
        ref = ForwardDiff.jacobian(dense_var, θ0)

        gmrf_var(θ) = var(GMRF(zeros(N), Qof(θ), LinearSolve.CHOLMODFactorization()))
        chordal_var(θ) = var(ChordalGMRF(zeros(N), Qof(θ)))
        workspace_var(θ) = var(WorkspaceGMRF(zeros(N), Qof(θ)))

        @test ForwardDiff.jacobian(gmrf_var, θ0) ≈ ref rtol = 1.0e-8
        @test ForwardDiff.jacobian(chordal_var, θ0) ≈ ref rtol = 1.0e-8
        @test ForwardDiff.jacobian(workspace_var, θ0) ≈ ref rtol = 1.0e-8
    end

    @testset "std differentiates through var" begin
        ref = ForwardDiff.gradient(θ -> sum(sqrt.(dense_var(θ))), θ0)
        gmrf_std(θ) = sum(std(GMRF(zeros(N), Qof(θ), LinearSolve.CHOLMODFactorization())))
        @test ForwardDiff.gradient(gmrf_std, θ0) ≈ ref rtol = 1.0e-8
    end

    @testset "ConstrainedGMRF constraint correction is differentiated" begin
        A = zeros(2, N)
        A[1, 1] = 1.0
        A[1, 2] = 1.0
        A[2, 5] = 1.0
        A[2, 9] = -2.0
        e = zeros(2)

        # Σ_c = Σ - ΣAᵀ(AΣAᵀ)⁻¹AΣ, computed densely.
        function dense_constrained_var(θ)
            Σ = inv(Matrix(Qof(θ)))
            M = Σ * A'
            return diag(Σ - M * ((A * M) \ M'))
        end

        constrained_sum(θ) = sum(
            var(
                ConstrainedGMRF(
                    GMRF(zeros(N), Qof(θ), LinearSolve.CHOLMODFactorization()), A, e
                )
            )
        )

        ref = ForwardDiff.gradient(θ -> sum(dense_constrained_var(θ)), θ0)
        @test constrained_sum(θ0) ≈ sum(dense_constrained_var(θ0)) rtol = 1.0e-10
        @test ForwardDiff.gradient(constrained_sum, θ0) ≈ ref rtol = 1.0e-8

        # Regression guard: differentiating only the base term yields the
        # *unconstrained* gradient, which is wrong but looks plausible.
        unconstrained_ref = ForwardDiff.gradient(θ -> sum(dense_var(θ)), θ0)
        @test ref ≉ unconstrained_ref
    end

    @testset "ChordalGMRF promotes a Float64 mean against a Dual precision" begin
        # `AbstractGMRF{T, L}` requires `L <: AbstractMatrix{T}`; without
        # promotion this construction throws a TypeError.
        g = ChordalGMRF(zeros(N), Qof(ForwardDiff.Dual.(θ0, (1.0, 0.0), (0.0, 1.0))))
        @test eltype(mean(g)) <: ForwardDiff.Dual
        @test length(var(g)) == N
    end

    @testset "diagonal precision" begin
        θd = [1.0, 2.0, 3.0]
        diag_sum(θ) = sum(var(GMRF(zeros(3), Diagonal(θ), LinearSolve.DiagonalFactorization())))
        @test ForwardDiff.gradient(diag_sum, θd) ≈ -1 ./ θd .^ 2 rtol = 1.0e-10
    end

    @testset "constant precision gives a zero gradient" begin
        # Differentiating only through the mean: `var` really is constant here, so
        # zero is the right answer rather than a dropped-partials artifact.
        Q0 = Qof(θ0)
        mean_only(θ) = sum(var(GMRF(θ[1] * ones(N), Q0, LinearSolve.CHOLMODFactorization())))
        @test ForwardDiff.gradient(mean_only, [1.0]) ≈ [0.0] atol = 1.0e-12
    end

    @testset "non-selinv solver throws rather than returning zeros" begin
        krylov_sum(θ) = sum(var(GMRF(zeros(N), Qof(θ), LinearSolve.KrylovJL_CG())))
        @test_throws ArgumentError ForwardDiff.gradient(krylov_sum, θ0)
    end

    @testset "primal values are unchanged" begin
        ref = sum(dense_var(θ0))
        Q = Qof(θ0)
        @test sum(var(GMRF(zeros(N), Q, LinearSolve.CHOLMODFactorization()))) ≈ ref rtol = 1.0e-10
        @test sum(var(ChordalGMRF(zeros(N), Q))) ≈ ref rtol = 1.0e-10
        @test sum(var(WorkspaceGMRF(zeros(N), Q))) ≈ ref rtol = 1.0e-10
    end
end
