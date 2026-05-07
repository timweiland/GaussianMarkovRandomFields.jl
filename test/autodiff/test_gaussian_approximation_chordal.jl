using Test
using GaussianMarkovRandomFields
using Distributions: logpdf, Poisson, Normal
using SparseArrays
using LinearAlgebra
using Random

using DifferentiationInterface
using FiniteDiff, Mooncake

chordal_backends = Any[("Mooncake", AutoMooncake())]

@testset "$backend_name ChordalGMRF autodiff tests" for (backend_name, backend) in chordal_backends
    # Set seed for reproducibility
    Random.seed!(42)
    fd_backend = AutoFiniteDiff()

    # Helper function to create simple AR(1) precision matrix
    function ar_precision(ρ, k)
        return spdiagm(-1 => -ρ * ones(k - 1), 0 => ones(k) .+ ρ^2, 1 => -ρ * ones(k - 1))
    end

    # Test pipeline: hyperparameters → ChordalGMRF → gaussian_approximation → logpdf
    function test_gauss_approx_pipeline_chordal(θ::Vector, y::Vector, x::Vector, k::Int)
        # Extract hyperparameters
        ρ = θ[1]        # AR parameter
        μ_const = θ[2]  # constant mean

        # Create precision matrix
        Q = ar_precision(ρ, k)

        # Create constant mean vector
        μ = μ_const * ones(k)

        # Create prior ChordalGMRF
        prior_gmrf = ChordalGMRF(μ, Q)

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
            θ -> test_gauss_approx_pipeline_chordal(θ, y, x, k),
            backend,
            θ
        )

        grad_fd = DifferentiationInterface.gradient(
            θ -> test_gauss_approx_pipeline_chordal(θ, y, x, k),
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
                    θ -> test_gauss_approx_pipeline_chordal(θ, y, x, k),
                    backend,
                    θ
                )

                grad_fd = DifferentiationInterface.gradient(
                    θ -> test_gauss_approx_pipeline_chordal(θ, y, x, k),
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

            prior_gmrf = ChordalGMRF(μ, Q)

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
            θ -> test_gauss_approx_pipeline_chordal(θ, y, x, k),
            backend,
            θ
        )

        grad_fd = DifferentiationInterface.gradient(
            θ -> test_gauss_approx_pipeline_chordal(θ, y, x, k),
            fd_backend,
            θ
        )

        abs_error = abs.(grad_test - grad_fd)
        rel_error = abs_error ./ (abs.(grad_fd) .+ 1.0e-10)

        @test maximum(abs_error) < 1.0e-3
        @test maximum(rel_error) < 5.0e-2
    end

    @testset "Basic logpdf autodiff" begin
        k = 10
        z = randn(k)

        function test_logpdf_pipeline(θ::AbstractVector, z::AbstractVector, k)
            ρ = θ[1]
            μ_const = θ[2]

            Q = ar_precision(ρ, k)
            μ = μ_const * ones(k)

            gmrf = ChordalGMRF(μ, Q)
            return logpdf(gmrf, z)
        end

        θ = [0.5, 0.1]

        # Compute gradients using AD backend
        grad_test = DifferentiationInterface.gradient(
            θ -> test_logpdf_pipeline(θ, z, k),
            backend,
            θ
        )

        # Compute gradients using finite differences
        grad_fd = DifferentiationInterface.gradient(
            θ -> test_logpdf_pipeline(θ, z, k),
            fd_backend,
            θ
        )

        # Check AD gradients match finite differences
        abs_error = abs.(grad_test - grad_fd)
        rel_error = abs_error ./ (abs.(grad_fd) .+ 1.0e-10)

        @test maximum(abs_error) < 1.0e-4
        @test maximum(rel_error) < 1.0e-2
    end

    @testset "2D grid precision" begin
        # Build 2D grid precision matrix using spdiagm (Zygote-compatible)
        function grid_precision(α, grid_size)
            n = grid_size^2

            # Main diagonal
            diag_main = fill(4.0 + α, n)

            # Horizontal neighbors (±1 diagonals), but skip row boundaries
            horiz = [-1.0 * (mod(i, grid_size) != 0) for i in 1:(n - 1)]

            # Vertical neighbors (±grid_size diagonals)
            vert = fill(-1.0, n - grid_size)

            return spdiagm(-grid_size => vert, -1 => horiz, 0 => diag_main, 1 => horiz, grid_size => vert)
        end

        grid_size = 4
        n = grid_size^2
        z = randn(n)

        function test_grid_pipeline(θ::AbstractVector, z::AbstractVector, grid_size)
            α = θ[1]
            μ_const = θ[2]

            Q = grid_precision(α, grid_size)
            n = grid_size^2
            μ = μ_const * ones(n)

            gmrf = ChordalGMRF(μ, Q)
            return logpdf(gmrf, z)
        end

        θ = [0.5, 0.1]

        grad_test = DifferentiationInterface.gradient(
            θ -> test_grid_pipeline(θ, z, grid_size),
            backend,
            θ
        )

        grad_fd = DifferentiationInterface.gradient(
            θ -> test_grid_pipeline(θ, z, grid_size),
            fd_backend,
            θ
        )

        abs_error = abs.(grad_test - grad_fd)
        rel_error = abs_error ./ (abs.(grad_fd) .+ 1.0e-10)

        @test maximum(abs_error) < 1.0e-4
        @test maximum(rel_error) < 1.0e-2
    end

    @testset "2D grid with Poisson gaussian_approximation" begin
        function grid_precision(α, grid_size)
            n = grid_size^2
            diag_main = fill(4.0 + α, n)
            horiz = [-1.0 * (mod(i, grid_size) != 0) for i in 1:(n - 1)]
            vert = fill(-1.0, n - grid_size)
            return spdiagm(-grid_size => vert, -1 => horiz, 0 => diag_main, 1 => horiz, grid_size => vert)
        end

        grid_size = 3
        n = grid_size^2
        y = [2, 1, 3, 2, 1, 4, 2, 1, 2]  # Poisson count data
        x = randn(n) .+ 0.5

        function test_grid_gauss_approx(θ, y, x, grid_size)
            α, μ_const = θ
            Q = grid_precision(α, grid_size)
            n = grid_size^2
            μ = μ_const * ones(n)

            prior_gmrf = ChordalGMRF(μ, Q)

            obs_model = ExponentialFamily(Poisson)
            obs_lik = obs_model(PoissonObservations(y))
            posterior_gmrf = gaussian_approximation(prior_gmrf, obs_lik)
            return logpdf(posterior_gmrf, x)
        end

        θ = [0.5, 0.3]

        grad_test = DifferentiationInterface.gradient(
            θ -> test_grid_gauss_approx(θ, y, x, grid_size),
            backend,
            θ
        )

        grad_fd = DifferentiationInterface.gradient(
            θ -> test_grid_gauss_approx(θ, y, x, grid_size),
            fd_backend,
            θ
        )

        abs_error = abs.(grad_test - grad_fd)
        rel_error = abs_error ./ (abs.(grad_fd) .+ 1.0e-10)

        @test maximum(abs_error) < 1.0e-2
        @test maximum(rel_error) < 5.0e-2
    end
end
