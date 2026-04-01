using GaussianMarkovRandomFields
using Distributions: logpdf
using SparseArrays
using LinearAlgebra
using Random

using DifferentiationInterface
using FiniteDiff, Zygote, ForwardDiff

function ar_precision_sparse(ρ, k)
    return spdiagm(-1 => -ρ * ones(k - 1), 0 => ones(k) .+ ρ^2, 1 => -ρ * ones(k - 1))
end

# Pipeline: θ → WorkspaceGMRF → logpdf
function test_workspace_logpdf_pipeline(θ::Vector, z::Vector, k::Int)
    ρ = θ[1]
    μ_const = θ[2]
    Q = ar_precision_sparse(ρ, k)
    μ = μ_const * ones(k)
    gmrf = WorkspaceGMRF(μ, Q)
    return logpdf(gmrf, z)
end

# Pipeline: θ → WorkspaceGMRF → gaussian_approximation → logpdf
function test_workspace_ga_pipeline(θ::Vector, y::Vector, x::Vector, k::Int)
    ρ = θ[1]
    μ_const = θ[2]
    Q = ar_precision_sparse(ρ, k)
    μ = μ_const * ones(k)
    prior = WorkspaceGMRF(μ, Q)
    obs_model = ExponentialFamily(Poisson)
    obs_lik = obs_model(PoissonObservations(y))
    posterior = gaussian_approximation(prior, obs_lik)
    return logpdf(posterior, x)
end

# Reference pipeline with standard GMRF (for cross-check)
function test_gmrf_logpdf_pipeline(θ::Vector, z::Vector, k::Int)
    ρ = θ[1]
    μ_const = θ[2]
    Q = ar_precision_sparse(ρ, k)
    μ = μ_const * ones(k)
    gmrf = GMRF(μ, Q)
    return logpdf(gmrf, z)
end

function test_gmrf_ga_pipeline(θ::Vector, y::Vector, x::Vector, k::Int)
    ρ = θ[1]
    μ_const = θ[2]
    Q = ar_precision_sparse(ρ, k)
    μ = μ_const * ones(k)
    prior = GMRF(μ, Q)
    obs_model = ExponentialFamily(Poisson)
    obs_lik = obs_model(PoissonObservations(y))
    posterior = gaussian_approximation(prior, obs_lik)
    return logpdf(posterior, x)
end

backends = Any[("Zygote", AutoZygote()), ("ForwardDiff", AutoForwardDiff())]

@testset "$backend_name WorkspaceGMRF autodiff" for (backend_name, backend) in backends
    Random.seed!(42)
    fd_backend = AutoFiniteDiff()

    @testset "logpdf gradient" begin
        k = 10
        θ = [0.5, 0.1]
        z = randn(k)

        grad_test = DifferentiationInterface.gradient(
            θ -> test_workspace_logpdf_pipeline(θ, z, k),
            backend, θ
        )
        grad_fd = DifferentiationInterface.gradient(
            θ -> test_workspace_logpdf_pipeline(θ, z, k),
            fd_backend, θ
        )

        abs_error = abs.(grad_test - grad_fd)
        rel_error = abs_error ./ (abs.(grad_fd) .+ 1.0e-10)
        @test maximum(abs_error) < 1.0e-4
        @test maximum(rel_error) < 1.0e-2
    end

    @testset "logpdf gradient matches GMRF" begin
        k = 10
        θ = [0.5, 0.1]
        z = randn(k)

        grad_ws = DifferentiationInterface.gradient(
            θ -> test_workspace_logpdf_pipeline(θ, z, k),
            backend, θ
        )
        grad_gmrf = DifferentiationInterface.gradient(
            θ -> test_gmrf_logpdf_pipeline(θ, z, k),
            backend, θ
        )

        @test grad_ws ≈ grad_gmrf rtol = 1.0e-8
    end

    @testset "GA + logpdf gradient" begin
        k = 8
        θ = [0.5, 0.0]
        y = [2, 1, 3, 0, 4, 1, 2, 3]
        x = zeros(k)

        grad_test = DifferentiationInterface.gradient(
            θ -> test_workspace_ga_pipeline(θ, y, x, k),
            backend, θ
        )
        grad_fd = DifferentiationInterface.gradient(
            θ -> test_workspace_ga_pipeline(θ, y, x, k),
            fd_backend, θ
        )

        abs_error = abs.(grad_test - grad_fd)
        rel_error = abs_error ./ (abs.(grad_fd) .+ 1.0e-10)
        @test maximum(abs_error) < 2.0e-2
        @test maximum(rel_error) < 5.0e-2
    end

    @testset "GA + logpdf gradient matches GMRF" begin
        k = 8
        θ = [0.5, 0.0]
        y = [2, 1, 3, 0, 4, 1, 2, 3]
        x = zeros(k)

        grad_ws = DifferentiationInterface.gradient(
            θ -> test_workspace_ga_pipeline(θ, y, x, k),
            backend, θ
        )
        grad_gmrf = DifferentiationInterface.gradient(
            θ -> test_gmrf_ga_pipeline(θ, y, x, k),
            backend, θ
        )

        @test grad_ws ≈ grad_gmrf rtol = 1.0e-4
    end
end
