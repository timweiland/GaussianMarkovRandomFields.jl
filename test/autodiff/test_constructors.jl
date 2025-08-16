using GaussianMarkovRandomFields
using SparseArrays
using LinearAlgebra
using Random

using Zygote
using Distributions

@testset "GMRF constructor autodiff tests" begin
    # Set seed for reproducibility
    Random.seed!(42)
    
    # Helper function to create simple AR(1) precision matrix
    function ar_precision(ρ, k)
        return spdiagm(-1 => -ρ * ones(k - 1), 0 => ones(k) .+ ρ^2, 1 => -ρ * ones(k - 1))
    end
    
    # Test pipeline: construct GMRF and access its properties
    function test_constructor_pipeline(θ::Vector, k::Int)
        # Extract hyperparameters
        ρ = θ[1]        # AR parameter
        μ_const = θ[2]  # constant mean
        
        # Create precision matrix
        Q = ar_precision(ρ, k)
        
        # Create constant mean vector
        μ = μ_const * ones(k)
        
        # Create GMRF - use default algorithm with our custom rrules
        gmrf = GMRF(μ, Q)
        
        # Return sum of mean (to test differentiation through construction)
        return sum(mean(gmrf))
    end
    
    # Compare AD gradients with finite differences
    function compare_constructor_gradients(θ::Vector, k::Int; h::Float64=1e-6)
        # Define objective function
        f(θ) = test_constructor_pipeline(θ, k)
        
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
    
    @testset "Constructor autodiff with default algorithm" begin
        k = 8
        θ = [0.5, 0.1]  # [ρ, μ_const]
        
        grad_zygote, grad_fd = compare_constructor_gradients(θ, k)
        
        # Check Zygote gradients match finite differences
        abs_error_zygote = abs.(grad_zygote - grad_fd)
        rel_error_zygote = abs_error_zygote ./ (abs.(grad_fd) .+ 1e-10)
        
        @test maximum(abs_error_zygote) < 1e-4
        @test maximum(rel_error_zygote) < 1e-2
    end
    
    @testset "Constructor autodiff with smaller system" begin
        k = 6
        θ = [0.3, -0.2]  # [ρ, μ_const]
        
        grad_zygote, grad_fd = compare_constructor_gradients(θ, k)
        
        # Check Zygote gradients match finite differences
        abs_error_zygote = abs.(grad_zygote - grad_fd)
        rel_error_zygote = abs_error_zygote ./ (abs.(grad_fd) .+ 1e-10)
        
        @test maximum(abs_error_zygote) < 1e-4
        @test maximum(rel_error_zygote) < 1e-2
    end
end
