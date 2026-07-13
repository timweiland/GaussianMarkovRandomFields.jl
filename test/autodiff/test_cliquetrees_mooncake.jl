using Test
using GaussianMarkovRandomFields
using Distributions: logpdf, var, Poisson, Normal
using SparseArrays
using LinearAlgebra
using LinearSolve
using Random

using DifferentiationInterface
using FiniteDiff, Mooncake, MooncakeSparse

@testset "Mooncake CliqueTrees-backed GMRF autodiff tests" begin
    Random.seed!(42)
    backend = AutoMooncake()
    fd_backend = AutoFiniteDiff()

    ar_precision_ct(ρ, k) =
        spdiagm(-1 => -ρ * ones(k - 1), 0 => ones(k) .+ ρ^2, 1 => -ρ * ones(k - 1))

    prior_ct(θ, k) = GMRF(θ[2] * ones(k), ar_precision_ct(θ[1], k), CliqueTreesFactorization())

    @testset "logpdf pipeline" begin
        k = 10
        z = randn(k)
        pipeline(θ) = logpdf(prior_ct(θ, k), z)

        for θ in ([0.5, 0.1], [0.3, -0.2])
            grad_mc = DifferentiationInterface.gradient(pipeline, backend, θ)
            grad_fd = DifferentiationInterface.gradient(pipeline, fd_backend, θ)
            @test grad_mc ≈ grad_fd atol = 1.0e-4 rtol = 1.0e-4
        end
    end

    @testset "var pipeline (selected inversion)" begin
        k = 8
        w = randn(k)
        pipeline(θ) = dot(w, var(prior_ct(θ, k)))

        θ = [0.4, 0.2]
        grad_mc = DifferentiationInterface.gradient(pipeline, backend, θ)
        grad_fd = DifferentiationInterface.gradient(pipeline, fd_backend, θ)
        @test grad_mc ≈ grad_fd atol = 1.0e-4 rtol = 1.0e-4
    end

    @testset "gaussian_approximation pipeline (Poisson)" begin
        k = 8
        y = [2, 1, 3, 2, 1, 4, 2, 1]
        x_eval = randn(k) .+ 0.5

        function pipeline(θ)
            obs_lik = ExponentialFamily(Poisson)(PoissonObservations(y))
            posterior = gaussian_approximation(prior_ct(θ, k), obs_lik)
            return logpdf(posterior, x_eval)
        end

        θ = [0.4, 0.5]
        grad_mc = DifferentiationInterface.gradient(pipeline, backend, θ)
        grad_fd = DifferentiationInterface.gradient(pipeline, fd_backend, θ)
        @test grad_mc ≈ grad_fd atol = 1.0e-3 rtol = 5.0e-2
    end

    @testset "gaussian_approximation pipeline (conjugate Normal)" begin
        k = 6
        y = randn(k) .* 0.3 .+ 0.2
        x_eval = randn(k)

        function pipeline(θ)
            obs_lik = ExponentialFamily(Normal)(y; σ = 0.3)
            posterior = gaussian_approximation(prior_ct(θ, k), obs_lik)
            return logpdf(posterior, x_eval)
        end

        θ = [0.3, 0.1]
        grad_mc = DifferentiationInterface.gradient(pipeline, backend, θ)
        grad_fd = DifferentiationInterface.gradient(pipeline, fd_backend, θ)
        @test grad_mc ≈ grad_fd atol = 1.0e-3 rtol = 5.0e-2
    end

    @testset "posterior variance objective (predictive uncertainty)" begin
        k = 6
        y = [1, 2, 1, 3, 1, 2]

        function pipeline(θ)
            obs_lik = ExponentialFamily(Poisson)(PoissonObservations(y))
            posterior = gaussian_approximation(prior_ct(θ, k), obs_lik)
            return sum(var(posterior))
        end

        θ = [0.4, 0.3]
        grad_mc = DifferentiationInterface.gradient(pipeline, backend, θ)
        grad_fd = DifferentiationInterface.gradient(pipeline, fd_backend, θ)
        @test grad_mc ≈ grad_fd atol = 1.0e-3 rtol = 5.0e-2
    end

    @testset "WorkspaceGMRF: logpdf with shared workspace" begin
        k = 10
        z = randn(k)
        ws = GMRFWorkspace(ar_precision_ct(0.5, k), CliqueTreesBackend)

        # One workspace reused across all gradient evaluations — the pattern
        # downstream INLA-style loops rely on.
        pipeline(θ) = logpdf(WorkspaceGMRF(θ[2] * ones(k), ar_precision_ct(θ[1], k), ws), z)

        for θ in ([0.5, 0.1], [0.3, -0.2])
            grad_mc = DifferentiationInterface.gradient(pipeline, backend, θ)
            grad_fd = DifferentiationInterface.gradient(pipeline, fd_backend, θ)
            @test grad_mc ≈ grad_fd atol = 1.0e-4 rtol = 1.0e-4
        end
    end

    @testset "WorkspaceGMRF: gaussian_approximation pipeline (Poisson)" begin
        k = 8
        y = [2, 1, 3, 2, 1, 4, 2, 1]
        x_eval = randn(k) .+ 0.5
        ws = GMRFWorkspace(ar_precision_ct(0.4, k), CliqueTreesBackend)

        function pipeline(θ)
            prior = WorkspaceGMRF(θ[2] * ones(k), ar_precision_ct(θ[1], k), ws)
            obs_lik = ExponentialFamily(Poisson)(PoissonObservations(y))
            posterior = gaussian_approximation(prior, obs_lik)
            return logpdf(posterior, x_eval)
        end

        θ = [0.4, 0.5]
        grad_mc = DifferentiationInterface.gradient(pipeline, backend, θ)
        grad_fd = DifferentiationInterface.gradient(pipeline, fd_backend, θ)
        @test grad_mc ≈ grad_fd atol = 1.0e-3 rtol = 5.0e-2
    end

    @testset "WorkspaceGMRF: Laplace marginal likelihood objective" begin
        k = 8
        y = [2, 1, 3, 2, 1, 4, 2, 1]
        ws = GMRFWorkspace(ar_precision_ct(0.4, k), CliqueTreesBackend)

        function pipeline(θ)
            prior = WorkspaceGMRF(θ[2] * ones(k), ar_precision_ct(θ[1], k), ws)
            obs_lik = ExponentialFamily(Poisson)(PoissonObservations(y))
            posterior = gaussian_approximation(prior, obs_lik)
            xs = mean(posterior)
            return logpdf(prior, xs) + loglik(xs, obs_lik) - logpdf(posterior, xs)
        end

        θ = [0.4, 0.5]
        grad_mc = DifferentiationInterface.gradient(pipeline, backend, θ)
        grad_fd = DifferentiationInterface.gradient(pipeline, fd_backend, θ)
        @test grad_mc ≈ grad_fd atol = 1.0e-3 rtol = 5.0e-2
    end

    @testset "WorkspaceGMRF: variance objective" begin
        k = 8
        w = randn(k)
        ws = GMRFWorkspace(ar_precision_ct(0.4, k), CliqueTreesBackend)

        pipeline(θ) = dot(w, var(WorkspaceGMRF(θ[2] * ones(k), ar_precision_ct(θ[1], k), ws)))

        θ = [0.4, 0.2]
        grad_mc = DifferentiationInterface.gradient(pipeline, backend, θ)
        grad_fd = DifferentiationInterface.gradient(pipeline, fd_backend, θ)
        @test grad_mc ≈ grad_fd atol = 1.0e-4 rtol = 1.0e-4
    end

    @testset "WorkspaceGMRF: friendly error for CHOLMOD backend" begin
        k = 6
        z = randn(k)
        ws_cholmod = GMRFWorkspace(ar_precision_ct(0.4, k))
        f_cholmod(θ) = logpdf(WorkspaceGMRF(θ[2] * ones(k), ar_precision_ct(θ[1], k), ws_cholmod), z)
        @test_throws Exception DifferentiationInterface.gradient(f_cholmod, backend, [0.4, 0.1])
    end

    @testset "ConstrainedGMRF: logpdf" begin
        k = 8
        A = ones(1, k)
        e = [0.0]
        z = randn(k)
        z .-= sum(z) / k

        pipeline(θ) = logpdf(
            ConstrainedGMRF(prior_ct(θ, k), A, e), z
        )

        for θ in ([0.5, 0.3], [0.3, -0.2])
            grad_mc = DifferentiationInterface.gradient(pipeline, backend, θ)
            grad_fd = DifferentiationInterface.gradient(pipeline, fd_backend, θ)
            @test grad_mc ≈ grad_fd atol = 1.0e-4 rtol = 1.0e-4
        end
    end

    @testset "ConstrainedGMRF: variance objective" begin
        k = 8
        A = ones(1, k)
        e = [0.0]
        w = randn(k)

        pipeline(θ) = dot(w, var(ConstrainedGMRF(prior_ct(θ, k), A, e)))

        θ = [0.4, 0.2]
        grad_mc = DifferentiationInterface.gradient(pipeline, backend, θ)
        grad_fd = DifferentiationInterface.gradient(pipeline, fd_backend, θ)
        @test grad_mc ≈ grad_fd atol = 1.0e-4 rtol = 1.0e-4
    end

    @testset "ConstrainedGMRF: gaussian_approximation" begin
        k = 8
        A = ones(1, k)
        e = [0.0]
        y = [2, 1, 3, 2, 1, 4, 2, 1]
        x_eval = randn(k)
        x_eval .-= sum(x_eval) / k

        function pipeline(θ)
            prior = ConstrainedGMRF(prior_ct(θ, k), A, e)
            obs_lik = ExponentialFamily(Poisson)(PoissonObservations(y))
            posterior = gaussian_approximation(prior, obs_lik)
            return logpdf(posterior, x_eval)
        end

        θ = [0.4, 0.5]
        grad_mc = DifferentiationInterface.gradient(pipeline, backend, θ)
        grad_fd = DifferentiationInterface.gradient(pipeline, fd_backend, θ)
        @test grad_mc ≈ grad_fd atol = 1.0e-3 rtol = 5.0e-2
    end

    @testset "ConstrainedGMRF: Laplace marginal likelihood" begin
        k = 8
        A = ones(1, k)
        e = [0.0]
        y = [2, 1, 3, 2, 1, 4, 2, 1]

        function pipeline(θ)
            prior = ConstrainedGMRF(prior_ct(θ, k), A, e)
            obs_lik = ExponentialFamily(Poisson)(PoissonObservations(y))
            posterior = gaussian_approximation(prior, obs_lik)
            xs = mean(posterior)
            return logpdf(prior, xs) + loglik(xs, obs_lik) - logpdf(posterior, xs)
        end

        θ = [0.4, 0.5]
        grad_mc = DifferentiationInterface.gradient(pipeline, backend, θ)
        grad_fd = DifferentiationInterface.gradient(pipeline, fd_backend, θ)
        @test grad_mc ≈ grad_fd atol = 1.0e-3 rtol = 5.0e-2
    end

    @testset "Gauss-Newton likelihood: actionable reverse-mode error" begin
        k = 2
        y_nlsq = [1.0, 0.5, 2.0]
        nlsq_lik = NonlinearLeastSquaresModel(
            x -> [x[1] + 2x[2], sin(x[1]), x[2]^2], k
        )(y_nlsq; σ = 0.3)
        ws = GMRFWorkspace(ar_precision_ct(0.4, k), CliqueTreesBackend)

        function pipeline(θ)
            prior = WorkspaceGMRF(θ[2] * ones(k), ar_precision_ct(θ[1], k), ws)
            return logpdf(gaussian_approximation(prior, nlsq_lik), zeros(k))
        end

        @test_throws "Reverse-mode automatic differentiation" DifferentiationInterface.gradient(
            pipeline, backend, [0.4, 0.1]
        )
    end

    @testset "Constrained WorkspaceGMRF: logpdf" begin
        k = 8
        # Sum-to-zero constraint; nonzero μ_const keeps the raw mean
        # infeasible, exercising the residual terms of the correction.
        A = ones(1, k)
        e = [0.0]
        z = randn(k)
        z .-= sum(z) / k
        ws = GMRFWorkspace(ar_precision_ct(0.4, k), CliqueTreesBackend)

        pipeline(θ) = logpdf(
            WorkspaceGMRF(θ[2] * ones(k), ar_precision_ct(θ[1], k), ws, A, e), z
        )

        for θ in ([0.5, 0.3], [0.3, -0.2])
            grad_mc = DifferentiationInterface.gradient(pipeline, backend, θ)
            grad_fd = DifferentiationInterface.gradient(pipeline, fd_backend, θ)
            @test grad_mc ≈ grad_fd atol = 1.0e-4 rtol = 1.0e-4
        end
    end

    @testset "Constrained WorkspaceGMRF: variance objective" begin
        k = 8
        A = ones(1, k)
        e = [0.0]
        w = randn(k)
        ws = GMRFWorkspace(ar_precision_ct(0.4, k), CliqueTreesBackend)

        pipeline(θ) = dot(
            w, var(WorkspaceGMRF(θ[2] * ones(k), ar_precision_ct(θ[1], k), ws, A, e))
        )

        θ = [0.4, 0.2]
        grad_mc = DifferentiationInterface.gradient(pipeline, backend, θ)
        grad_fd = DifferentiationInterface.gradient(pipeline, fd_backend, θ)
        @test grad_mc ≈ grad_fd atol = 1.0e-4 rtol = 1.0e-4
    end

    @testset "Constrained WorkspaceGMRF: gaussian_approximation" begin
        k = 8
        A = ones(1, k)
        e = [0.0]
        y = [2, 1, 3, 2, 1, 4, 2, 1]
        x_eval = randn(k)
        x_eval .-= sum(x_eval) / k
        ws = GMRFWorkspace(ar_precision_ct(0.4, k), CliqueTreesBackend)

        function pipeline(θ)
            prior = WorkspaceGMRF(θ[2] * ones(k), ar_precision_ct(θ[1], k), ws, A, e)
            obs_lik = ExponentialFamily(Poisson)(PoissonObservations(y))
            posterior = gaussian_approximation(prior, obs_lik)
            return logpdf(posterior, x_eval)
        end

        θ = [0.4, 0.5]
        grad_mc = DifferentiationInterface.gradient(pipeline, backend, θ)
        grad_fd = DifferentiationInterface.gradient(pipeline, fd_backend, θ)
        @test grad_mc ≈ grad_fd atol = 1.0e-3 rtol = 5.0e-2
    end

    @testset "Constrained WorkspaceGMRF: Laplace marginal likelihood" begin
        k = 8
        A = ones(1, k)
        e = [0.0]
        y = [2, 1, 3, 2, 1, 4, 2, 1]
        ws = GMRFWorkspace(ar_precision_ct(0.4, k), CliqueTreesBackend)

        function pipeline(θ)
            prior = WorkspaceGMRF(θ[2] * ones(k), ar_precision_ct(θ[1], k), ws, A, e)
            obs_lik = ExponentialFamily(Poisson)(PoissonObservations(y))
            posterior = gaussian_approximation(prior, obs_lik)
            xs = mean(posterior)
            return logpdf(prior, xs) + loglik(xs, obs_lik) - logpdf(posterior, xs)
        end

        θ = [0.4, 0.5]
        grad_mc = DifferentiationInterface.gradient(pipeline, backend, θ)
        grad_fd = DifferentiationInterface.gradient(pipeline, fd_backend, θ)
        @test grad_mc ≈ grad_fd atol = 1.0e-3 rtol = 5.0e-2
    end
end
