using GaussianMarkovRandomFields
using SparseArrays
using LinearAlgebra
using Random
using LinearSolve

using DifferentiationInterface
using Enzyme, FiniteDiff, Zygote, ForwardDiff
using Distributions
using Test
using ReTest

# Helper to create precision matrix
function ar_precision(ρ, k)
    return spdiagm(-1 => -ρ * ones(k - 1), 0 => ones(k) .+ ρ^2, 1 => -ρ * ones(k - 1))
end

# Test GMRF(μ, Q) constructor gradients
function test_constructor_basic(θ::Vector, k::Int)
    ρ = θ[1]
    μ_val = θ[2]

    Q = ar_precision(ρ, k)
    μ = μ_val * ones(k)

    gmrf = GMRF(μ, Q)

    # Return something that depends on GMRF internals
    # Avoid logdetcov for Zygote compatibility
    return sum(mean(gmrf)) + sum(gmrf.precision)
end

# Test GMRF(μ, Q, factorization) constructor gradients
function test_constructor_fact(θ::Vector, k::Int)
    ρ = θ[1]
    μ_val = θ[2]

    Q = ar_precision(ρ, k)
    μ = μ_val * ones(k)

    gmrf = GMRF(μ, Q, LinearSolve.CHOLMODFactorization())

    return logdetcov(gmrf)
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

macro run_basic_constructor_tests(backend)
    return quote
        # Set seed for reproducibility
        Random.seed!(42)

        @testset "Basic constructor gradients" begin
            k = 10
            θ = [0.5, 0.1]

            grad_test = DifferentiationInterface.gradient(
                θ -> test_constructor_basic(θ, k),
                $backend,
                θ
            )

            grad_fd = DifferentiationInterface.gradient(
                θ -> test_constructor_basic(θ, k),
                AutoFiniteDiff(),
                θ
            )

            abs_error = abs.(grad_test - grad_fd)
            rel_error = abs_error ./ (abs.(grad_fd) .+ 1.0e-10)

            @test maximum(abs_error) < 1.0e-4
            @test maximum(rel_error) < 1.0e-2
        end
    end
end

macro run_factorized_constructor_tests(backend)
    return quote
        # Set seed for reproducibility
        Random.seed!(42)

        @testset "Factorized constructor gradients" begin
            k = 8
            θ = [0.3, -0.2]

            grad_test = DifferentiationInterface.gradient(
                θ -> test_constructor_fact(θ, k),
                $backend,
                θ
            )

            grad_fd = DifferentiationInterface.gradient(
                θ -> test_constructor_fact(θ, k),
                AutoFiniteDiff(),
                θ
            )

            abs_error = abs.(grad_test - grad_fd)
            rel_error = abs_error ./ (abs.(grad_fd) .+ 1.0e-10)

            @test maximum(abs_error) < 1.0e-4
            @test maximum(rel_error) < 1.0e-2
        end
    end
end

macro run_symtridiagonal_tests(backend)
    return quote
        # Set seed for reproducibility
        Random.seed!(42)

        @testset "SymTridiagonal + LDLtFactorization constructor" begin
            k = 8
            θ = [1.5, 0.6, 0.2]  # [τ, ρ, μ_const]

            grad_test = DifferentiationInterface.gradient(
                θ -> test_constructor_pipeline_symtridiag(θ, k),
                $backend,
                θ
            )

            grad_fd = DifferentiationInterface.gradient(
                θ -> test_constructor_pipeline_symtridiag(θ, k),
                AutoFiniteDiff(),
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
                        $backend,
                        θ
                    )

                    grad_fd = DifferentiationInterface.gradient(
                        θ -> test_constructor_pipeline_symtridiag(θ, k),
                        AutoFiniteDiff(),
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
end

@testset_macro @run_basic_constructor_tests
@testset_macro @run_factorized_constructor_tests
@testset_macro @run_symtridiagonal_tests

@testset "ForwardDiff GMRF constructor autodiff tests" begin
    @run_basic_constructor_tests(AutoForwardDiff())
    @run_factorized_constructor_tests(AutoForwardDiff())
end

@testset "Zygote GMRF constructor autodiff tests" begin
    @run_basic_constructor_tests(AutoZygote())
end

if get(ENV, "GMRF_TEST_ENZYME", "false") == "true"
    @testset "Enzyme GMRF constructor autodiff tests" begin
        @run_basic_constructor_tests(AutoEnzyme(; function_annotation = Enzyme.Const))
        @run_factorized_constructor_tests(AutoEnzyme(; function_annotation = Enzyme.Const))
    end
end

@testset "ForwardDiff GMRF constructor autodiff tests (SymTridiagonal)" begin
    @run_symtridiagonal_tests(AutoForwardDiff())
end

# Skip Zygote for SymTridiagonal tests as it involves LDLtFactorization which Zygote might not support
@testset "Zygote GMRF constructor autodiff tests (SymTridiagonal)" begin
    @run_symtridiagonal_tests(AutoZygote())
end

if get(ENV, "GMRF_TEST_ENZYME", "false") == "true"
    @testset "Enzyme GMRF constructor autodiff tests (SymTridiagonal)" begin
        @run_symtridiagonal_tests(AutoEnzyme(; function_annotation = Enzyme.Const))
    end
end
