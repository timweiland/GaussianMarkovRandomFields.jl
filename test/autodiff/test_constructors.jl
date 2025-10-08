using GaussianMarkovRandomFields
using SparseArrays
using LinearAlgebra
using Random
using LinearSolve

using DifferentiationInterface
using Enzyme, FiniteDiff, Zygote
using Distributions

@testset "$backend_name GMRF constructor autodiff tests" for (backend_name, backend) in [("ForwardDiff", AutoForwardDiff()), ("Zygote", AutoZygote()), ("Enzyme", AutoEnzyme(; function_annotation = Enzyme.Const))]
    # Set seed for reproducibility
    Random.seed!(42)
    fd_backend = AutoFiniteDiff()

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
        gmrf = GMRF(μ, Q, LinearSolve.CHOLMODFactorization())

        return sum(mean(gmrf)) + sum(Q)
    end

    # Test pipeline with SymTridiagonal + LDLtFactorization
    function test_constructor_pipeline_symtridiag(θ::Vector, k::Int)
        # Extract hyperparameters
        τ = θ[1]        # precision parameter
        ρ = θ[2]        # AR correlation
        μ_const = θ[3]  # constant mean

        # Create AR1 model which produces SymTridiagonal
        model = AR1Model(k)
        Q = precision_matrix(model; τ = τ, ρ = ρ)

        # Create constant mean vector
        μ = μ_const * ones(k)

        # Create GMRF with LDLtFactorization
        gmrf = GMRF(μ, Q, LinearSolve.LDLtFactorization())

        return sum(mean(gmrf)) + sum(Q)
    end

    @testset "Constructor autodiff with default algorithm" begin
        k = 8
        θ = [0.5, 0.1]  # [ρ, μ_const]

        grad_test = DifferentiationInterface.gradient(
            θ -> test_constructor_pipeline(θ, k),
            backend,
            θ
        )

        grad_fd = DifferentiationInterface.gradient(
            θ -> test_constructor_pipeline(θ, k),
            fd_backend,
            θ
        )

        abs_error = abs.(grad_test - grad_fd)
        rel_error = abs_error ./ (abs.(grad_fd) .+ 1.0e-10)

        @test maximum(abs_error) < 1.0e-4
        @test maximum(rel_error) < 1.0e-2
    end

    @testset "Constructor autodiff with smaller system" begin
        k = 6
        θ = [0.3, -0.2]  # [ρ, μ_const]

        grad_test = DifferentiationInterface.gradient(
            θ -> test_constructor_pipeline(θ, k),
            backend,
            θ
        )

        grad_fd = DifferentiationInterface.gradient(
            θ -> test_constructor_pipeline(θ, k),
            fd_backend,
            θ
        )

        abs_error = abs.(grad_test - grad_fd)
        rel_error = abs_error ./ (abs.(grad_fd) .+ 1.0e-10)

        @test maximum(abs_error) < 1.0e-4
        @test maximum(rel_error) < 1.0e-2
    end

    @testset "SymTridiagonal + LDLtFactorization constructor" begin
        k = 8
        θ = [1.5, 0.6, 0.2]  # [τ, ρ, μ_const]

        grad_test = DifferentiationInterface.gradient(
            θ -> test_constructor_pipeline_symtridiag(θ, k),
            backend,
            θ
        )

        grad_fd = DifferentiationInterface.gradient(
            θ -> test_constructor_pipeline_symtridiag(θ, k),
            fd_backend,
            θ
        )

        abs_error = abs.(grad_test - grad_fd)
        rel_error = abs_error ./ (abs.(grad_fd) .+ 1.0e-10)

        @test maximum(abs_error) < 1.0e-4
        @test maximum(rel_error) < 1.0e-2
    end

    @testset "SymTridiagonal constructor with different parameters" begin
        k = 6

        # Test different τ and ρ values
        for τ in [0.5, 2.0]
            for ρ in [0.3, 0.7]
                θ = [τ, ρ, 0.1]

                grad_test = DifferentiationInterface.gradient(
                    θ -> test_constructor_pipeline_symtridiag(θ, k),
                    backend,
                    θ
                )

                grad_fd = DifferentiationInterface.gradient(
                    θ -> test_constructor_pipeline_symtridiag(θ, k),
                    fd_backend,
                    θ
                )

                abs_error = abs.(grad_test - grad_fd)
                rel_error = abs_error ./ (abs.(grad_fd) .+ 1.0e-10)

                @test maximum(abs_error) < 1.0e-4
                @test maximum(rel_error) < 1.0e-2
            end
        end
    end
end
