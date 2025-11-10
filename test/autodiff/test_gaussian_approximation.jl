using GaussianMarkovRandomFields
using Distributions: logpdf
using SparseArrays
using LinearAlgebra
using Random
using LinearSolve

using DifferentiationInterface
using Enzyme, FiniteDiff, Zygote

@testset "$backend_name gaussian_approximation autodiff tests" for (backend_name, backend) in [("Zygote", AutoZygote()), ("Enzyme", AutoEnzyme(; function_annotation = Enzyme.Const))]
    # Set seed for reproducibility
    Random.seed!(42)
    fd_backend = AutoFiniteDiff()

    # Helper function to create simple AR(1) precision matrix
    function ar_precision(ρ, k)
        return spdiagm(-1 => -ρ * ones(k - 1), 0 => ones(k) .+ ρ^2, 1 => -ρ * ones(k - 1))
    end

    # Test pipeline: hyperparameters → GMRF → gaussian_approximation → logpdf
    function test_gauss_approx_pipeline(θ::Vector, y::Vector, x::Vector, k::Int)
        # Extract hyperparameters
        ρ = θ[1]        # AR parameter
        μ_const = θ[2]  # constant mean

        # Create precision matrix
        Q = ar_precision(ρ, k)

        # Create constant mean vector
        μ = μ_const * ones(k)

        # Create prior GMRF
        prior_gmrf = GMRF(μ, Q, LinearSolve.CHOLMODFactorization())

        # Create Poisson observation likelihood
        obs_model = ExponentialFamily(Poisson)
        poisson_obs = PoissonObservations(y)
        obs_lik = obs_model(poisson_obs)

        # Find Gaussian approximation
        posterior_gmrf = gaussian_approximation(prior_gmrf, obs_lik)

        # Compute logpdf at evaluation point x
        return logpdf(posterior_gmrf, x)
    end

    # Test pipeline with SymTridiagonal + LDLtFactorization
    function test_gauss_approx_pipeline_symtridiag(θ::Vector, y::Vector, x::Vector, k::Int)
        # Extract hyperparameters
        τ = θ[1]        # precision parameter
        ρ = θ[2]        # AR correlation
        μ_const = θ[3]  # constant mean

        # Create AR1 model which produces SymTridiagonal
        model = AR1Model(k)
        Q = precision_matrix(model; τ = τ, ρ = ρ)

        # Create constant mean vector
        μ = μ_const * ones(k)

        # Create prior GMRF with LDLtFactorization
        prior_gmrf = GMRF(μ, Q, LinearSolve.LDLtFactorization())

        # Create Poisson observation likelihood
        obs_model = ExponentialFamily(Poisson)
        poisson_obs = PoissonObservations(y)
        obs_lik = obs_model(poisson_obs)

        # Find Gaussian approximation
        posterior_gmrf = gaussian_approximation(prior_gmrf, obs_lik)

        # Compute logpdf at evaluation point x
        return logpdf(posterior_gmrf, x)
    end

    @testset "Poisson likelihood with gaussian_approximation" begin
        k = 8
        θ = [0.4, 0.5]  # [ρ, μ_const]
        y = [2, 1, 3, 2, 1, 4, 2, 1]  # Poisson count data
        x = randn(k) .+ 0.5  # Evaluation point

        grad_test = DifferentiationInterface.gradient(
            θ -> test_gauss_approx_pipeline(θ, y, x, k),
            backend,
            θ
        )

        grad_fd = DifferentiationInterface.gradient(
            θ -> test_gauss_approx_pipeline(θ, y, x, k),
            fd_backend,
            θ
        )

        abs_error = abs.(grad_test - grad_fd)
        rel_error = abs_error ./ (abs.(grad_fd) .+ 1.0e-10)

        @test maximum(abs_error) < 1.0e-3
        @test maximum(rel_error) < 5.0e-2
    end

    @testset "Different hyperparameter values" begin
        k = 6
        y = [1, 2, 1, 3, 1, 2]
        x = randn(k) .+ 0.3

        # Test different ρ and μ values
        for ρ in [0.2, 0.5]
            for μ_const in [0.3, 0.8]
                θ = [ρ, μ_const]

                grad_test = DifferentiationInterface.gradient(
                    θ -> test_gauss_approx_pipeline(θ, y, x, k),
                    backend,
                    θ
                )

                grad_fd = DifferentiationInterface.gradient(
                    θ -> test_gauss_approx_pipeline(θ, y, x, k),
                    fd_backend,
                    θ
                )

                abs_error = abs.(grad_test - grad_fd)
                rel_error = abs_error ./ (abs.(grad_fd) .+ 1.0e-10)

                @test maximum(abs_error) < 2.0e-2  # Relaxed for complex optimization
                @test maximum(rel_error) < 5.0e-2
            end
        end
    end

    @testset "Gaussian (conjugate) likelihood" begin
        # Test with Gaussian likelihood - should also work through rrule
        k = 6
        θ = [0.3, 0.1]
        y = randn(k) .* 0.3 .+ 0.2
        x = randn(k)

        function gaussian_lik_pipeline(θ, y, x, k)
            ρ, μ_const = θ
            Q = ar_precision(ρ, k)
            μ = μ_const * ones(k)
            prior_gmrf = GMRF(μ, Q, LinearSolve.CHOLMODFactorization())
            obs_model = ExponentialFamily(Normal)
            obs_lik = obs_model(y; σ = 0.5)
            posterior_gmrf = gaussian_approximation(prior_gmrf, obs_lik)
            return logpdf(posterior_gmrf, x)
        end

        grad_test = DifferentiationInterface.gradient(
            θ -> gaussian_lik_pipeline(θ, y, x, k),
            backend,
            θ
        )

        grad_fd = DifferentiationInterface.gradient(
            θ -> gaussian_lik_pipeline(θ, y, x, k),
            fd_backend,
            θ
        )

        abs_error = abs.(grad_test - grad_fd)
        rel_error = abs_error ./ (abs.(grad_fd) .+ 1.0e-10)

        @test maximum(abs_error) < 1.0e-3
        @test maximum(rel_error) < 5.0e-2
    end

    @testset "Small system" begin
        # Test with very small system
        k = 4
        θ = [0.6, 0.4]
        y = [1, 2, 1, 1]
        x = randn(k) .+ 0.4

        grad_test = DifferentiationInterface.gradient(
            θ -> test_gauss_approx_pipeline(θ, y, x, k),
            backend,
            θ
        )

        grad_fd = DifferentiationInterface.gradient(
            θ -> test_gauss_approx_pipeline(θ, y, x, k),
            fd_backend,
            θ
        )

        abs_error = abs.(grad_test - grad_fd)
        rel_error = abs_error ./ (abs.(grad_fd) .+ 1.0e-10)

        @test maximum(abs_error) < 1.0e-3
        @test maximum(rel_error) < 5.0e-2
    end

    @testset "SymTridiagonal + LDLtFactorization with Poisson" begin
        k = 8
        θ = [1.5, 0.6, 0.5]  # [τ, ρ, μ_const]
        y = [2, 1, 3, 2, 1, 4, 2, 1]
        x = randn(k) .+ 0.5

        grad_test = DifferentiationInterface.gradient(
            θ -> test_gauss_approx_pipeline_symtridiag(θ, y, x, k),
            backend,
            θ
        )

        grad_fd = DifferentiationInterface.gradient(
            θ -> test_gauss_approx_pipeline_symtridiag(θ, y, x, k),
            fd_backend,
            θ
        )

        abs_error = abs.(grad_test - grad_fd)
        rel_error = abs_error ./ (abs.(grad_fd) .+ 1.0e-10)

        @test maximum(abs_error) < 1.0e-3
        @test maximum(rel_error) < 5.0e-2
    end

    @testset "SymTridiagonal with different parameters" begin
        k = 6
        y = [1, 2, 1, 3, 1, 2]
        x = randn(k) .+ 0.3

        # Test different τ and ρ values
        for τ in [0.5, 2.0]
            for ρ in [0.3, 0.7]
                θ = [τ, ρ, 0.4]

                grad_test = DifferentiationInterface.gradient(
                    θ -> test_gauss_approx_pipeline_symtridiag(θ, y, x, k),
                    backend,
                    θ
                )

                grad_fd = DifferentiationInterface.gradient(
                    θ -> test_gauss_approx_pipeline_symtridiag(θ, y, x, k),
                    fd_backend,
                    θ
                )

                abs_error = abs.(grad_test - grad_fd)
                rel_error = abs_error ./ (abs.(grad_fd) .+ 1.0e-10)

                @test maximum(abs_error) < 2.0e-2
                @test maximum(rel_error) < 5.0e-2
            end
        end
    end
end
