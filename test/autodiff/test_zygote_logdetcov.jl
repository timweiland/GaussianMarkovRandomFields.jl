using GaussianMarkovRandomFields
using Distributions: logpdf, Poisson
using SparseArrays
using LinearAlgebra
using LinearSolve
using Random
using Statistics: var, std

using DifferentiationInterface
using FiniteDiff, Zygote, ForwardDiff
using ReTest: @testset, @test

# Regression tests for the log-determinant term of the Zygote gradient.
#
# `logpdf` decomposes into `logdetcov + sqmahal`. Before the `logdetcov` rrules
# existed, Zygote traced into the stored Cholesky factorization, which carries no
# tangent, and silently dropped the whole log-determinant contribution — giving a
# gradient that was ~14% off rather than an error.
#
# These tests MUST use a precision matrix with genuine Cholesky fill-in. A
# tridiagonal AR(1) precision has none, and the pattern-projection subtleties in
# the selected inverse are invisible without it — which is why the pre-existing
# AD test matrix missed this bug.

# 2D grid Laplacian: non-chordal, so its Cholesky factor fills in.
function grid_laplacian(m)
    rows, cols = Int[], Int[]
    for j in 1:m, i in 1:m
        idx = (j - 1) * m + i
        if i < m
            push!(rows, idx)
            push!(cols, idx + 1)
        end
        if j < m
            push!(rows, idx)
            push!(cols, idx + m)
        end
    end
    W = sparse([rows; cols], [cols; rows], 1.0, m * m, m * m)
    return spdiagm(0 => vec(sum(W, dims = 2))) - W
end

@testset "Zygote logdetcov gradients (precision with fill-in)" begin
    grid_size = 8
    L = grid_laplacian(grid_size)
    n = size(L, 1)

    rng = Random.MersenneTwister(42)
    z = randn(rng, n)
    y = rand(rng, Poisson(3), n)

    # θ = [log scale, log nugget]; both hyperparameters enter Q, and θ[2] also
    # sets the mean, so a dropped term cannot hide in an unused coordinate.
    Qof(θ) = sparse(exp(θ[1]) * (L + exp(θ[2]) * I))
    μof(θ) = θ[2] * ones(n)
    θ = [log(2.0), 0.3]

    @testset "test matrix genuinely fills in" begin
        # Guards the tests below: swapping in a chordal (e.g. tridiagonal)
        # precision would silently defeat their purpose.
        Q = Qof(θ)
        @test nnz(sparse(cholesky(Q).L)) > nnz(tril(Q))
    end

    # Dense, package-independent references: ForwardDiff over dense linear
    # algebra never touches the sparse factorization or any rrule under test.
    dense_logdetcov(θ) = -logdet(Matrix(Qof(θ)))
    function dense_logpdf(θ)
        Q = Matrix(Qof(θ))
        r = z .- μof(θ)
        return -0.5 * (n * log(2π) - logdet(Q) + dot(r, Q, r))
    end

    ref_logdetcov = ForwardDiff.gradient(dense_logdetcov, θ)
    ref_logpdf = ForwardDiff.gradient(dense_logpdf, θ)

    @testset "dense references agree with finite differences" begin
        @test ref_logdetcov ≈ FiniteDiff.finite_difference_gradient(dense_logdetcov, θ) rtol = 1.0e-5
        @test ref_logpdf ≈ FiniteDiff.finite_difference_gradient(dense_logpdf, θ) rtol = 1.0e-5
    end

    # A GMRFWorkspace is a reusable symbolic factorization, so it is built once
    # here rather than inside the differentiated function: Zygote cannot trace
    # CHOLMOD's symbolic phase, and constructing one inside the closure raises
    # regardless of which rules exist. Sharing it across the calls below also
    # exercises the pullback's `ensure_loaded!`, since the workspace gets
    # refactorized at a different Q in between.
    ws = GMRFWorkspace(Qof(θ))

    gmrf_constructors = Any[
        ("ChordalGMRF", θ -> ChordalGMRF(μof(θ), Qof(θ))),
        ("GMRF/CHOLMOD", θ -> GMRF(μof(θ), Qof(θ), LinearSolve.CHOLMODFactorization())),
        ("WorkspaceGMRF", θ -> WorkspaceGMRF(μof(θ), Qof(θ), ws)),
    ]

    @testset "$name" for (name, mk) in gmrf_constructors
        @testset "logdetcov" begin
            g = DifferentiationInterface.gradient(θ -> logdetcov(mk(θ)), AutoZygote(), θ)
            @test g ≈ ref_logdetcov rtol = 1.0e-8
        end

        @testset "logpdf" begin
            g = DifferentiationInterface.gradient(θ -> logpdf(mk(θ), z), AutoZygote(), θ)
            @test g ≈ ref_logpdf rtol = 1.0e-8

            # The original bug returned exactly the gradient with the
            # log-determinant term missing. Pin that specific failure mode.
            dropped = ref_logpdf .+ 0.5 .* ref_logdetcov
            @test !isapprox(g, dropped, rtol = 1.0e-3)
        end

        @testset "gaussian_approximation -> logpdf" begin
            obs_lik = ExponentialFamily(Poisson)(PoissonObservations(y))
            f(θ) = logpdf(gaussian_approximation(mk(θ), obs_lik), z)

            g = DifferentiationInterface.gradient(f, AutoZygote(), θ)
            ref = DifferentiationInterface.gradient(f, AutoForwardDiff(), θ)

            @test g ≈ ref rtol = 1.0e-6
            # ForwardDiff and Zygote share no rrule code here, but cross-check
            # against finite differences too so a shared-primal bug cannot pass.
            @test g ≈ FiniteDiff.finite_difference_gradient(f, θ) rtol = 1.0e-5
        end
    end

    @testset "var and std refuse instead of returning a wrong number" begin
        # No solver makes `var` differentiable under Zygote, so the rule refuses
        # unconditionally. The point is the message: without it Zygote fails
        # somewhere in its own internals with an error that names neither the
        # operation nor the GMRF type.
        for (name, mk) in gmrf_constructors
            @testset "$name" begin
                for (op, f) in (
                        ("var", θ -> sum(var(mk(θ)))),
                        ("std", θ -> sum(std(mk(θ)))),
                    )
                    err = try
                        DifferentiationInterface.gradient(f, AutoZygote(), θ)
                        nothing
                    catch e
                        e
                    end
                    @test err !== nothing
                    msg = sprint(showerror, err)
                    @test occursin("var", msg)
                    @test occursin("ForwardDiff", msg)
                end
            end
        end

        # The primal is untouched — only differentiation refuses.
        @test var(GMRF(μof(θ), Qof(θ), LinearSolve.CHOLMODFactorization())) ≈
            diag(inv(Matrix(Qof(θ)))) rtol = 1.0e-8
    end

    @testset "constrained WorkspaceGMRF" begin
        # Sum-to-zero constraint, the usual identifiability constraint on
        # RW/Besag-style models.
        A = ones(1, n)
        e = [0.0]

        mk(θ) = WorkspaceGMRF(μof(θ), Qof(θ), ws, A, e)

        @testset "logdetcov ignores the constraint" begin
            # logdetcov is the *base* log-determinant; the Rue-Held correction is
            # a separate term that only logpdf adds. So the constrained gradient
            # must equal the plain -logdet(Q) reference. If the rule ever grew a
            # constraint contribution, this is what would catch it.
            g = DifferentiationInterface.gradient(θ -> logdetcov(mk(θ)), AutoZygote(), θ)
            @test g ≈ ref_logdetcov rtol = 1.0e-8

            unconstrained =
                DifferentiationInterface.gradient(
                θ -> logdetcov(WorkspaceGMRF(μof(θ), Qof(θ), ws)),
                AutoZygote(), θ
            )
            @test g ≈ unconstrained rtol = 1.0e-10
        end

        @testset "logpdf still carries the constraint correction" begin
            # Guards the other direction: adding the logdetcov rule must not
            # disturb the existing constrained logpdf rrule, whose gradient does
            # depend on the constraint through log_constraint_correction.
            z_feasible = z .- (sum(z) / n)
            f(θ) = logpdf(mk(θ), z_feasible)

            g = DifferentiationInterface.gradient(f, AutoZygote(), θ)
            @test g ≈ FiniteDiff.finite_difference_gradient(f, θ) rtol = 1.0e-5

            # The correction genuinely moves the gradient, so the constrained and
            # unconstrained logpdf gradients must differ — otherwise this test
            # would pass even if the constraint were being ignored entirely.
            plain(θ) = logpdf(WorkspaceGMRF(μof(θ), Qof(θ), ws), z_feasible)
            @test !isapprox(
                g, DifferentiationInterface.gradient(plain, AutoZygote(), θ), rtol = 1.0e-3
            )
        end
    end
end
