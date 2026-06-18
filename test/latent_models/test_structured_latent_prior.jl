using ReTest: @testset, @test
using GaussianMarkovRandomFields
using ForwardDiff   # activates the DI ForwardDiff backend the factor assembler uses
# Loading these activates the SparseADLikelihoods extension, so factor-group Hessian sparsity is
# detected structurally (SparseConnectivityTracer) rather than assumed dense.
using SparseConnectivityTracer, SparseMatrixColorings
using SparseArrays
using LinearAlgebra
using Random
using Distributions: Normal, MvNormal, logpdf

# Quadratic-drift AR model with analytic derivatives as ground truth (same density as the
# AutoDiffLatentPrior test): x[1] ~ N(0, 1/τ), x[t] ~ N(a·x[t-1]², 1/τ). The Hessian depends on x
# (genuinely non-Gaussian) and couples adjacent components.
function _sqd_logp(x, τ, a)
    n = length(x)
    s = x[1]^2
    @inbounds for t in 2:n
        s += (x[t] - a * x[t - 1]^2)^2
    end
    return 0.5 * n * log(τ) - 0.5 * n * log(2π) - 0.5 * τ * s
end

function _sqd_grad(x, τ, a)
    n = length(x)
    g = zeros(n)
    g[1] = -τ * x[1]
    @inbounds for t in 2:n
        r = x[t] - a * x[t - 1]^2
        g[t - 1] += τ * r * 2 * a * x[t - 1]
        g[t] += -τ * r
    end
    return g
end

function _sqd_neg_hessian(x, τ, a)
    n = length(x)
    rows = Int[]; cols = Int[]; vals = Float64[]
    push!(rows, 1); push!(cols, 1); push!(vals, τ)
    @inbounds for t in 2:n
        r = x[t] - a * x[t - 1]^2
        push!(rows, t); push!(cols, t); push!(vals, τ)
        push!(rows, t - 1); push!(cols, t - 1); push!(vals, 0.5 * τ * (-4 * a * r + 8 * a^2 * x[t - 1]^2))
        doff = 0.5 * τ * (-4 * a * x[t - 1])
        push!(rows, t - 1); push!(cols, t); push!(vals, doff)
        push!(rows, t); push!(cols, t - 1); push!(vals, doff)
    end
    return sparse(rows, cols, vals, n, n)
end

@testset "StructuredLatentPrior" begin

    @testset "local_quadratic / prior_logdensity match analytic (quadratic-drift)" begin
        Random.seed!(1)
        n = 5; τ = 1.5; a = 0.4
        # One factor per conditional: an initial N(0, 1/τ) and the nonlinear transitions.
        g_init = LatentFactorGroup([(1,)], (vals, θ) -> logpdf(Normal(0.0, 1 / sqrt(θ.τ)), vals[1]))
        g_trans = LatentFactorGroup(
            [(t, t - 1) for t in 2:n],
            (vals, θ) -> logpdf(Normal(θ.a * vals[2]^2, 1 / sqrt(θ.τ)), vals[1]),
        )
        # Tridiagonal pattern: the structural Hessian sparsity of the model.
        rows = Int[]; cols = Int[]
        for i in 1:n
            push!(rows, i); push!(cols, i)
        end
        for t in 2:n
            push!(rows, t); push!(cols, t - 1); push!(rows, t - 1); push!(cols, t)
        end
        pat = SparseMatrixCSC{Bool, Int}(sparse(rows, cols, trues(length(rows)), n, n))

        prior = StructuredLatentPrior(n, (g_init, g_trans), pat; hyperparams = (:τ, :a))

        for _ in 1:5
            x = randn(n) .* 0.5
            lq = local_quadratic(prior, x; τ = τ, a = a)
            @test lq.Q isa SparseMatrixCSC
            @test Matrix(lq.Q) ≈ Matrix(_sqd_neg_hessian(x, τ, a)) atol = 1.0e-8
            @test lq.h ≈ _sqd_grad(x, τ, a) .+ _sqd_neg_hessian(x, τ, a) * x atol = 1.0e-8
            @test lq.logp_ref ≈ _sqd_logp(x, τ, a) atol = 1.0e-10
            @test prior_logdensity(prior, x; τ = τ, a = a) ≈ _sqd_logp(x, τ, a) atol = 1.0e-10
        end
    end

    @testset "structural sparsity: diagonal-covariance block factors scatter sparsely" begin
        # x[1:2] ~ N(0, I); x[3:4] ~ N(x[1:2], I). The diagonal covariances mean there is no
        # within-block coupling (x[1]↔x[2], x[3]↔x[4]) and no off-diagonal cross coupling
        # (x[3]↔x[2]) — only x[a]↔x[a+2]. So a block factor's Hessian is genuinely sparse, and
        # the pattern must NOT need the dense K×K block.
        Random.seed!(7)
        n = 4
        g1 = LatentFactorGroup([(1, 2)], (vals, θ) -> logpdf(MvNormal(zeros(2), 1.0 * I(2)), vals))
        g2 = LatentFactorGroup([(3, 4, 1, 2)], (vals, θ) -> logpdf(MvNormal(vals[3:4], 1.0 * I(2)), vals[1:2]))

        # Analytic joint precision Q (all blocks diagonal): Q[1:2,1:2]=2I, Q[3:4,3:4]=I,
        # Q[1:2,3:4]=Q[3:4,1:2]=-I.
        Qref = zeros(4, 4)
        Qref[1, 1] = 2.0; Qref[2, 2] = 2.0; Qref[3, 3] = 1.0; Qref[4, 4] = 1.0
        Qref[1, 3] = Qref[3, 1] = -1.0; Qref[2, 4] = Qref[4, 2] = -1.0

        # Sparse pattern with only the real couplings (no x[1]↔x[2], x[3]↔x[4], x[3]↔x[2], …).
        rows = [1, 2, 3, 4, 1, 3, 2, 4]; cols = [1, 2, 3, 4, 3, 1, 4, 2]
        pat_sparse = SparseMatrixCSC{Bool, Int}(sparse(rows, cols, trues(length(rows)), n, n))
        prior_sparse = StructuredLatentPrior(n, (g1, g2), pat_sparse)

        # Dense pattern: the extra entries are structural zeros and must scatter to 0.
        pat_dense = SparseMatrixCSC{Bool, Int}(sparse(trues(n, n)))
        prior_dense = StructuredLatentPrior(n, (g1, g2), pat_dense)

        for _ in 1:4
            x = randn(n)
            lqs = local_quadratic(prior_sparse, x)
            lqd = local_quadratic(prior_dense, x)
            @test Matrix(lqs.Q) ≈ Qref atol = 1.0e-10
            @test Matrix(lqd.Q) ≈ Qref atol = 1.0e-10
            @test nnz(lqs.Q) < nnz(lqd.Q)   # the sparse pattern genuinely drops the structural zeros
        end
    end

end
