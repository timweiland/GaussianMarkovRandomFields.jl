using GaussianMarkovRandomFields
using Distributions: logpdf
using SparseArrays
using LinearAlgebra
using Random
using LinearSolve

using DifferentiationInterface
using Enzyme, FiniteDiff, Zygote

@testset "$backend_name logpdf autodiff tests" for (backend_name, backend) in [("ForwardDiff", AutoForwardDiff()), ("Zygote", AutoZygote()), ("Enzyme", AutoEnzyme(; function_annotation = Enzyme.Const))]
    # Set seed for reproducibility
    Random.seed!(42)
    fd_backend = AutoFiniteDiff()

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
        gmrf = GMRF(μ, Q, LinearSolve.CHOLMODFactorization())

        # Compute logpdf
        return logpdf(gmrf, z)
    end

    # Test pipeline with SymTridiagonal + LDLtFactorization
    function test_pipeline_symtridiag(θ::Vector, z::Vector, k::Int)
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

        # Compute logpdf
        return logpdf(gmrf, z)
    end

    @testset "Default algorithm logpdf autodiff" begin
        k = 10
        θ = [0.5, 0.1]  # [ρ, μ_const]
        z = randn(k)

        # Compute gradients using AD backend
        grad_test = DifferentiationInterface.gradient(
            θ -> test_pipeline(θ, z, k),
            backend,
            θ
        )

        # Compute gradients using finite differences
        grad_fd = DifferentiationInterface.gradient(
            θ -> test_pipeline(θ, z, k),
            fd_backend,
            θ
        )

        # Check AD gradients match finite differences
        abs_error = abs.(grad_test - grad_fd)
        rel_error = abs_error ./ (abs.(grad_fd) .+ 1.0e-10)

        @test maximum(abs_error) < 1.0e-4
        @test maximum(rel_error) < 1.0e-2
    end

    @testset "Smaller system logpdf autodiff" begin
        k = 8
        θ = [0.3, -0.2]  # [ρ, μ_const]
        z = randn(k)

        grad_test = DifferentiationInterface.gradient(
            θ -> test_pipeline(θ, z, k),
            backend,
            θ
        )

        grad_fd = DifferentiationInterface.gradient(
            θ -> test_pipeline(θ, z, k),
            fd_backend,
            θ
        )

        abs_error = abs.(grad_test - grad_fd)
        rel_error = abs_error ./ (abs.(grad_fd) .+ 1.0e-10)

        @test maximum(abs_error) < 1.0e-4
        @test maximum(rel_error) < 1.0e-2
    end

    @testset "Different parameter values" begin
        k = 6
        z = randn(k)

        # Test different ρ values
        for ρ in [0.1, 0.5, 0.8]
            θ = [ρ, 0.0]

            grad_test = DifferentiationInterface.gradient(
                θ -> test_pipeline(θ, z, k),
                backend,
                θ
            )

            grad_fd = DifferentiationInterface.gradient(
                θ -> test_pipeline(θ, z, k),
                fd_backend,
                θ
            )

            abs_error = abs.(grad_test - grad_fd)
            rel_error = abs_error ./ (abs.(grad_fd) .+ 1.0e-10)

            @test maximum(abs_error) < 1.0e-4
            @test maximum(rel_error) < 1.0e-2
        end
    end

    @testset "SymTridiagonal + LDLtFactorization" begin
        k = 8
        θ = [1.5, 0.6, 0.2]  # [τ, ρ, μ_const]
        z = randn(k)

        grad_test = DifferentiationInterface.gradient(
            θ -> test_pipeline_symtridiag(θ, z, k),
            backend,
            θ
        )

        grad_fd = DifferentiationInterface.gradient(
            θ -> test_pipeline_symtridiag(θ, z, k),
            fd_backend,
            θ
        )

        abs_error = abs.(grad_test - grad_fd)
        rel_error = abs_error ./ (abs.(grad_fd) .+ 1.0e-10)

        @test maximum(abs_error) < 1.0e-4
        @test maximum(rel_error) < 1.0e-2
    end

    @testset "SymTridiagonal with different parameters" begin
        k = 6
        z = randn(k)

        # Test different τ and ρ values
        for τ in [0.5, 2.0]
            for ρ in [0.3, 0.7]
                θ = [τ, ρ, 0.1]

                grad_test = DifferentiationInterface.gradient(
                    θ -> test_pipeline_symtridiag(θ, z, k),
                    backend,
                    θ
                )

                grad_fd = DifferentiationInterface.gradient(
                    θ -> test_pipeline_symtridiag(θ, z, k),
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

@testset "Precision gradient computation tests" begin
    using GaussianMarkovRandomFields: compute_precision_gradient

    @testset "SymTridiagonal precision gradient" begin
        # Test compute_precision_gradient with SymTridiagonal matrix
        n = 5
        dv = [2.0, 3.0, 2.5, 3.5, 2.8]
        ev = [0.5, 0.6, 0.55, 0.62]
        Qinv = SymTridiagonal(dv, ev)
        r = [0.1, -0.2, 0.3, -0.1, 0.15]
        ȳ = 2.0

        grad = compute_precision_gradient(Qinv, r, ȳ)

        # Should return SymTridiagonal
        @test grad isa SymTridiagonal

        # Check diagonal elements: 0.5 * ȳ * (Qinv.dv - r .* r)
        @test grad.dv ≈ 0.5 .* ȳ .* (Qinv.dv .- r .* r)

        # Check off-diagonal: 0.5 * ȳ * (Qinv.ev - r[1:n-1] .* r[2:n])
        @test grad.ev ≈ 0.5 .* ȳ .* (Qinv.ev .- r[1:(n - 1)] .* r[2:n])
    end

    @testset "Sparse precision gradient" begin
        # Test compute_precision_gradient with SparseMatrixCSC
        using SparseArrays

        n = 4
        Qinv = sparse(
            [
                1.0 0.5 0.0 0.0;
                0.5 2.0 0.3 0.0;
                0.0 0.3 1.5 0.4;
                0.0 0.0 0.4 1.8
            ]
        )
        r = [0.2, -0.1, 0.15, -0.05]
        ȳ = 1.5

        grad = compute_precision_gradient(Qinv, r, ȳ)

        # Should return sparse matrix
        @test grad isa SparseMatrixCSC

        # Verify it has same sparsity pattern
        @test nnz(grad) == nnz(Qinv)

        # Check a few values
        @test grad[1, 1] ≈ 0.5 * ȳ * (Qinv[1, 1] - r[1]^2)
        @test grad[1, 2] ≈ 0.5 * ȳ * (Qinv[1, 2] - r[1] * r[2])
    end

    @testset "Symmetric sparse precision gradient" begin
        # Test compute_precision_gradient with Symmetric{SparseMatrixCSC}
        using SparseArrays, LinearAlgebra

        n = 3
        data = sparse(
            [
                2.0 0.5 0.0;
                0.5 3.0 0.6;
                0.0 0.6 2.5
            ]
        )
        Qinv = Symmetric(data)
        r = [0.1, -0.15, 0.2]
        ȳ = 1.0

        grad = compute_precision_gradient(Qinv, r, ȳ)

        # Should return Symmetric sparse
        @test grad isa Symmetric
        @test grad.data isa SparseMatrixCSC

        # Check symmetry is preserved
        @test grad[1, 2] ≈ grad[2, 1]
        @test grad[2, 3] ≈ grad[3, 2]
    end

    @testset "Generic fallback with warning" begin
        # Test generic fallback path (should trigger warning)
        # Use a dense Matrix which will use the generic method
        Qinv_dense = [2.0 0.5; 0.5 3.0]
        r = [0.1, -0.2]
        ȳ = 1.0

        # Should warn about using generic fallback
        @test_logs (:warn, r"Using generic fallback") compute_precision_gradient(Qinv_dense, r, ȳ)

        # Should still compute correct result
        grad = compute_precision_gradient(Qinv_dense, r, ȳ)
        expected = 0.5 .* ȳ .* (Qinv_dense .- r * r')
        @test grad ≈ expected
    end
end
