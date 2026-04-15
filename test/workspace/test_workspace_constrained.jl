using GaussianMarkovRandomFields
using GaussianMarkovRandomFields: has_constraints
using Distributions
using LinearAlgebra
using SparseArrays
using Random

using DifferentiationInterface
using FiniteDiff, Zygote

@testset "Constrained WorkspaceGMRF" begin

    @testset "Basic constrained GMRF - matches ConstrainedGMRF" begin
        n = 10
        Q = spdiagm(0 => 2.0 * ones(n), 1 => fill(-0.5, n - 1), -1 => fill(-0.5, n - 1))
        μ = randn(n)
        A = ones(1, n)  # sum-to-zero
        e = [0.0]

        ref = ConstrainedGMRF(GMRF(μ, Q), A, e)
        ws_constrained = WorkspaceGMRF(μ, Q, GMRFWorkspace(Q), A, e)

        @test ws_constrained isa WorkspaceGMRF
        @test has_constraints(ws_constrained)
        @test length(ws_constrained) == n
        @test mean(ws_constrained) ≈ mean(ref) rtol = 1.0e-8
        @test sum(mean(ws_constrained)) ≈ 0.0 atol = 1.0e-8
    end

    @testset "Variance matches ConstrainedGMRF" begin
        n = 10
        Q = spdiagm(0 => 2.0 * ones(n), 1 => fill(-0.5, n - 1), -1 => fill(-0.5, n - 1))
        μ = zeros(n)
        A = ones(1, n)
        e = [0.0]

        ref = ConstrainedGMRF(GMRF(μ, Q), A, e)
        ws_constrained = WorkspaceGMRF(μ, Q, GMRFWorkspace(Q), A, e)

        @test var(ws_constrained) ≈ var(ref) rtol = 1.0e-8
        @test all(var(ws_constrained) .>= 0)
    end

    @testset "logpdf matches ConstrainedGMRF" begin
        n = 10
        Q = spdiagm(0 => 2.0 * ones(n), 1 => fill(-0.5, n - 1), -1 => fill(-0.5, n - 1))
        μ = zeros(n)
        A = ones(1, n)
        e = [0.0]

        ref = ConstrainedGMRF(GMRF(μ, Q), A, e)
        ws_constrained = WorkspaceGMRF(μ, Q, GMRFWorkspace(Q), A, e)

        z = mean(ref)
        @test logpdf(ws_constrained, z) ≈ logpdf(ref, z) rtol = 1.0e-8
    end

    @testset "Sampling satisfies constraints" begin
        n = 8
        Q = spdiagm(0 => 2.0 * ones(n), 1 => fill(-0.5, n - 1), -1 => fill(-0.5, n - 1))
        μ = zeros(n)
        A = ones(1, n)
        e = [0.0]

        ws_constrained = WorkspaceGMRF(μ, Q, GMRFWorkspace(Q), A, e)

        rng = Random.MersenneTwister(42)
        for _ in 1:20
            sample = rand(rng, ws_constrained)
            @test sum(sample) ≈ 0.0 atol = 1.0e-8
        end
    end

    @testset "Multiple constraints" begin
        n = 6
        Q = spdiagm(0 => 2.0 * ones(n))
        μ = zeros(n)
        A = [1.0 1.0 1.0 1.0 1.0 1.0; 1.0 -1.0 0.0 0.0 0.0 0.0]
        e = [0.0, 0.0]

        ref = ConstrainedGMRF(GMRF(μ, Q), A, e)
        ws_constrained = WorkspaceGMRF(μ, Q, GMRFWorkspace(Q), A, e)

        @test mean(ws_constrained) ≈ mean(ref) rtol = 1.0e-8
        @test A * mean(ws_constrained) ≈ e atol = 1.0e-8
    end

    @testset "Constrained GA - Poisson matches reference" begin
        n = 8
        Q_prior = spdiagm(0 => ones(n))
        μ_prior = zeros(n)
        A = ones(1, n)
        e = [0.0]

        ref_base = GMRF(μ_prior, Q_prior)
        ref_constrained = ConstrainedGMRF(ref_base, A, e)
        obs_model = ExponentialFamily(Distributions.Poisson)
        y = PoissonObservations([2, 1, 3, 0, 4, 1, 2, 3])
        obs_lik = obs_model(y)
        ref_result = gaussian_approximation(ref_constrained, obs_lik)

        ws = GMRFWorkspace(Q_prior)
        ws_constrained = WorkspaceGMRF(μ_prior, Q_prior, ws, A, e)
        ws_result = gaussian_approximation(ws_constrained, obs_lik)

        @test ws_result isa WorkspaceGMRF
        @test has_constraints(ws_result)
        @test mean(ws_result) ≈ mean(ref_result) rtol = 1.0e-6
        @test sum(mean(ws_result)) ≈ 0.0 atol = 1.0e-6
    end

    @testset "Constrained GA - Bernoulli matches reference" begin
        n = 4
        Q_prior = spdiagm(0 => ones(n))
        μ_prior = zeros(n)
        A = ones(1, n)
        e = [0.0]

        ref_constrained = ConstrainedGMRF(GMRF(μ_prior, Q_prior), A, e)
        obs_model = ExponentialFamily(Distributions.Bernoulli)
        y = [1, 0, 1, 0]
        obs_lik = obs_model(y)
        ref_result = gaussian_approximation(ref_constrained, obs_lik; max_iter = 20)

        ws = GMRFWorkspace(Q_prior)
        ws_constrained = WorkspaceGMRF(μ_prior, Q_prior, ws, A, e)
        ws_result = gaussian_approximation(ws_constrained, obs_lik; max_iter = 20)

        @test ws_result isa WorkspaceGMRF
        @test has_constraints(ws_result)
        @test mean(ws_result) ≈ mean(ref_result) rtol = 1.0e-6
        @test sum(mean(ws_result)) ≈ 0.0 atol = 1.0e-6
    end

    @testset "Constrained logpdf autodiff" begin
        # Tests the gradient path through log_constraint_correction in the
        # WorkspaceGMRF logpdf rrule (src/workspace/autodiff.jl). Cross-checks
        # against AutoFiniteDiff and against the proven ConstrainedGMRF rrule.
        #
        # Note: Zygote only — the ForwardDiff extension does not yet override
        # the constrained WorkspaceGMRF constructor for Dual inputs.

        function ar_precision_sparse(ρ, k)
            return spdiagm(
                -1 => -ρ * ones(k - 1),
                0 => ones(k) .+ ρ^2,
                1 => -ρ * ones(k - 1),
            )
        end

        # Pipeline through WorkspaceGMRF — workspace is captured as a closure
        # so we don't pull cholesky! into Zygote's view (the constructor rrule
        # treats `ws` as NoTangent).
        function ws_constrained_pipeline(θ, z, k, A, e, ws)
            ρ, μ_const = θ[1], θ[2]
            Q = ar_precision_sparse(ρ, k)
            μ = μ_const * ones(k)
            return logpdf(WorkspaceGMRF(μ, Q, ws, A, e), z)
        end

        # Reference pipeline through ConstrainedGMRF
        function ref_constrained_pipeline(θ, z, k, A, e)
            ρ, μ_const = θ[1], θ[2]
            Q = ar_precision_sparse(ρ, k)
            μ = μ_const * ones(k)
            return logpdf(ConstrainedGMRF(GMRF(μ, Q), A, e), z)
        end

        Random.seed!(42)

        @testset "Single sum-to-zero constraint" begin
            k = 8
            A = ones(1, k)
            e = [0.0]
            θ = [0.5, 0.1]
            # z that satisfies the constraint
            z = randn(k); z .-= sum(z) / k
            # Build workspace once with an arbitrary Q matching the sparsity pattern
            ws = GMRFWorkspace(ar_precision_sparse(0.5, k))

            grad_ws_zyg = DifferentiationInterface.gradient(
                θ -> ws_constrained_pipeline(θ, z, k, A, e, ws), AutoZygote(), θ
            )
            grad_fd = DifferentiationInterface.gradient(
                θ -> ws_constrained_pipeline(θ, z, k, A, e, ws), AutoFiniteDiff(), θ
            )
            grad_ref = DifferentiationInterface.gradient(
                θ -> ref_constrained_pipeline(θ, z, k, A, e), AutoZygote(), θ
            )

            @test maximum(abs.(grad_ws_zyg - grad_fd)) < 1.0e-4
            @test grad_ws_zyg ≈ grad_ref rtol = 1.0e-6
        end

        @testset "Multiple constraints with non-zero e" begin
            k = 6
            A = [1.0 1.0 1.0 1.0 1.0 1.0; 1.0 -1.0 0.0 0.0 0.0 0.0]
            e = [0.5, -0.2]
            θ = [0.4, 0.3]
            # z that satisfies the constraints
            z_unc = randn(k)
            z = z_unc + A' * ((A * A') \ (e - A * z_unc))
            ws = GMRFWorkspace(ar_precision_sparse(0.4, k))

            grad_ws_zyg = DifferentiationInterface.gradient(
                θ -> ws_constrained_pipeline(θ, z, k, A, e, ws), AutoZygote(), θ
            )
            grad_fd = DifferentiationInterface.gradient(
                θ -> ws_constrained_pipeline(θ, z, k, A, e, ws), AutoFiniteDiff(), θ
            )
            grad_ref = DifferentiationInterface.gradient(
                θ -> ref_constrained_pipeline(θ, z, k, A, e), AutoZygote(), θ
            )

            @test maximum(abs.(grad_ws_zyg - grad_fd)) < 1.0e-4
            @test grad_ws_zyg ≈ grad_ref rtol = 1.0e-6
        end
    end
end
