using GaussianMarkovRandomFields
using LinearAlgebra, SparseArrays, LinearSolve, Random
using Distributions: logpdf, mean, var, std, logdetcov, MvNormal, Poisson
import Distributions
import CliqueTrees
import Zygote
import FiniteDiff

# 2D lattice precision — small but with genuine Cholesky fill-in.
function _ct_lattice_precision(m)
    n = m * m
    idx(i, j) = (j - 1) * m + i
    I_, J_, V_ = Int[], Int[], Float64[]
    for i in 1:m, j in 1:m
        push!(I_, idx(i, j)); push!(J_, idx(i, j)); push!(V_, 4.5)
        for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))
            ii, jj = i + di, j + dj
            if 1 <= ii <= m && 1 <= jj <= m
                push!(I_, idx(i, j)); push!(J_, idx(ii, jj)); push!(V_, -1.0)
            end
        end
    end
    return sparse(I_, J_, V_, n, n)
end

@testset "CliqueTrees GMRF backend" begin
    rng = MersenneTwister(2026)
    m = 6
    Q = _ct_lattice_precision(m)
    n = size(Q, 1)
    μ = randn(rng, n)
    Q_dense = Matrix(Q)
    Σ_dense = inv(Q_dense)

    gmrf = GMRF(μ, Q, CliqueTreesFactorization())

    @testset "Capability detection" begin
        @test GaussianMarkovRandomFields.supports_selinv(gmrf.linsolve_cache.alg) == Val{true}()
        @test GaussianMarkovRandomFields.supports_backward_solve(gmrf.linsolve_cache.alg) == Val{true}()
    end

    @testset "Mean from information vector" begin
        gmrf_info = GMRF(InformationVector(Q * μ), Q, CliqueTreesFactorization())
        @test mean(gmrf_info) ≈ μ rtol = 1.0e-10
    end

    @testset "logdetcov" begin
        @test logdetcov(gmrf) ≈ logdet(Σ_dense) rtol = 1.0e-10
    end

    @testset "Marginal variances (selinv diagonal)" begin
        @test var(gmrf) ≈ diag(Σ_dense) rtol = 1.0e-10
        @test std(gmrf) ≈ sqrt.(diag(Σ_dense)) rtol = 1.0e-10
    end

    @testset "Full selected inverse" begin
        Σ_sel = GaussianMarkovRandomFields.selinv(gmrf.linsolve_cache)
        @test Σ_sel isa Symmetric{Float64, <:SparseMatrixCSC}
        rows, cols, _ = findnz(Q)
        @test all(Σ_sel[i, j] ≈ Σ_dense[i, j] for (i, j) in zip(rows, cols))
    end

    @testset "Backward solve" begin
        # backward_solve maps z ↦ P⁻¹U⁻¹z with UᵀU = PQPᵀ, so applying it to
        # all identity columns gives B with B*Bᵀ = Q⁻¹ — the sampling property.
        B = hcat(
            [
                GaussianMarkovRandomFields.backward_solve(
                        gmrf.linsolve_cache, Vector{Float64}(I(n)[:, j])
                    ) for j in 1:n
            ]...
        )
        @test B * B' ≈ Σ_dense rtol = 1.0e-8
    end

    @testset "Sampling" begin
        Random.seed!(rng, 7)
        samples = hcat([rand(rng, gmrf) for _ in 1:5000]...)
        @test vec(sum(samples, dims = 2) / 5000) ≈ μ atol = 0.15
        emp_var = vec(sum(abs2, samples .- μ, dims = 2) / 5000)
        @test emp_var ≈ diag(Σ_dense) rtol = 0.2
    end

    @testset "logpdf" begin
        z = randn(rng, n)
        reference = MvNormal(μ, Symmetric(Σ_dense))
        @test logpdf(gmrf, z) ≈ logpdf(reference, z) rtol = 1.0e-10
    end

    @testset "Custom elimination ordering" begin
        # Passing a non-default ordering is broken upstream: LinearSolve's
        # LinearSolveCliqueTreesExt.makefactor unwraps `alg.alg` twice
        # (LinearSolve ≤ 3.87). Tripwire so we notice when the fix lands.
        @test_broken (GMRF(μ, Q, CliqueTreesFactorization(alg = CliqueTrees.AMD())); true)
    end

    @testset "Zygote hyperparameter gradients" begin
        k = 10
        z_obs = randn(rng, k)

        function ct_pipeline(θ)
            ρ = θ[1]
            Q_ar = spdiagm(-1 => -ρ * ones(k - 1), 0 => ones(k) .+ ρ^2, 1 => -ρ * ones(k - 1))
            x = GMRF(θ[2] * ones(k), Q_ar, CliqueTreesFactorization())
            return logpdf(x, z_obs)
        end

        θ0 = [0.5, 0.1]
        grad_zy = Zygote.gradient(ct_pipeline, θ0)[1]
        grad_fd = FiniteDiff.finite_difference_gradient(ct_pipeline, θ0)
        @test grad_zy ≈ grad_fd atol = 1.0e-6
    end

    @testset "gaussian_approximation carries the backend" begin
        y = rand.(rng, Distributions.Poisson.(exp.(clamp.(μ, -2.0, 2.0))))
        obs_lik = ExponentialFamily(Distributions.Poisson)(PoissonObservations(y))

        post_ct = gaussian_approximation(gmrf, obs_lik)
        post_ch = gaussian_approximation(GMRF(μ, Q, LinearSolve.CHOLMODFactorization()), obs_lik)

        @test post_ct.linsolve_cache.alg isa CliqueTreesFactorization
        @test mean(post_ct) ≈ mean(post_ch) rtol = 1.0e-8
        @test var(post_ct) ≈ var(post_ch) rtol = 1.0e-8
        @test logdetcov(post_ct) ≈ logdetcov(post_ch) rtol = 1.0e-8
    end
end
