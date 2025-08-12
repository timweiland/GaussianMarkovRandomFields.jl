using Test
using GaussianMarkovRandomFields
using Distributions: logpdf
using SparseArrays
using LinearAlgebra
using Random

using Zygote

@testset "Autodiff pipeline tests" begin
    # Set seed for reproducibility
    Random.seed!(42)
    
    # Helper function to create simple AR(1) precision matrix
    function ar_precision(ρ, k)
        return spdiagm(-1 => -ρ * ones(k - 1), 0 => ones(k) .+ ρ^2, 1 => -ρ * ones(k - 1))
    end
    
    # Test pipeline: hyperparameters → GMRF → logpdf
    function test_pipeline(θ::Vector, z::Vector, k::Int)
        # Extract hyperparameters
        ρ = θ[1]        # AR parameter
        μ_const = θ[2]  # constant mean
        
        # Create precision matrix
        Q = ar_precision(ρ, k)
        
        # Create constant mean vector
        μ = μ_const * ones(k)
        
        # Create GMRF with default algorithm
        gmrf = GMRF(μ, Q)
        
        # Compute logpdf
        return logpdf(gmrf, z)
    end
    
    # Compare AD gradients with finite differences
    function compare_gradients(θ::Vector, z::Vector, k::Int; h::Float64=1e-6)
        # Define objective function
        f(θ) = test_pipeline(θ, z, k)
        
        # Compute gradients using Zygote
        grad_zygote = Zygote.gradient(f, θ)[1]
        
        # Compute gradients using finite differences
        grad_fd = similar(θ)
        for i in 1:length(θ)
            θ_plus = copy(θ)
            θ_minus = copy(θ)
            θ_plus[i] += h
            θ_minus[i] -= h
            grad_fd[i] = (f(θ_plus) - f(θ_minus)) / (2h)
        end
        
        return grad_zygote, grad_fd
    end
    
    @testset "Default algorithm logpdf autodiff" begin
        k = 10
        θ = [0.5, 0.1]  # [ρ, μ_const]
        z = randn(k)
        
        grad_zygote, grad_fd = compare_gradients(θ, z, k)
        
        # Check Zygote gradients match finite differences
        abs_error_zygote = abs.(grad_zygote - grad_fd)
        rel_error_zygote = abs_error_zygote ./ (abs.(grad_fd) .+ 1e-10)
        
        @test maximum(abs_error_zygote) < 1e-4
        @test maximum(rel_error_zygote) < 1e-2
    end
    
    @testset "Smaller system logpdf autodiff" begin
        k = 8
        θ = [0.3, -0.2]  # [ρ, μ_const]
        z = randn(k)
        
        grad_zygote, grad_fd = compare_gradients(θ, z, k)
        
        # Check Zygote gradients match finite differences
        abs_error_zygote = abs.(grad_zygote - grad_fd)
        rel_error_zygote = abs_error_zygote ./ (abs.(grad_fd) .+ 1e-10)
        
        @test maximum(abs_error_zygote) < 1e-4
        @test maximum(rel_error_zygote) < 1e-2
    end
    
    @testset "Different parameter values" begin
        k = 6
        z = randn(k)
        
        # Test different ρ values
        for ρ in [0.1, 0.5, 0.8]
            θ = [ρ, 0.0]
            grad_zygote, grad_fd = compare_gradients(θ, z, k)
            
            # Check Zygote gradients match finite differences
            abs_error_zygote = abs.(grad_zygote - grad_fd)
            rel_error_zygote = abs_error_zygote ./ (abs.(grad_fd) .+ 1e-10)
            
            @test maximum(abs_error_zygote) < 1e-4
            @test maximum(rel_error_zygote) < 1e-2
        end
    end
end
