using GaussianMarkovRandomFields
using Test
using LinearAlgebra
using SparseArrays

@testset "BesagModel" begin
    @testset "Constructor" begin
        # Valid triangle graph
        W = sparse(Bool[0 1 1; 1 0 1; 1 1 0])
        model = BesagModel(W)
        @test model.adjacency == W
        @test model.regularization == 1.0e-5

        # Matrix input should be converted to sparse
        W_dense = Bool[0 1; 1 0]
        model_dense = BesagModel(W_dense)
        @test model_dense.adjacency isa SparseMatrixCSC

        # Custom regularization
        model_custom = BesagModel(W; regularization = 1.0e-4)
        @test model_custom.regularization == 1.0e-4

        # Invalid adjacency matrices
        @test_throws ArgumentError BesagModel([1 0; 0 1; 1 0])  # Not square
        @test_throws ArgumentError BesagModel([0 1; 0 0])  # Not symmetric
        @test_throws ArgumentError BesagModel([1 1; 1 0])  # Non-zero diagonal
        @test_throws ArgumentError BesagModel([0 0; 0 0])  # Isolated nodes
        @test_throws ArgumentError BesagModel(W; regularization = 0.0)  # Bad regularization
    end

    @testset "Hyperparameters" begin
        W = sparse(Bool[0 1; 1 0])
        model = BesagModel(W)
        params = hyperparameters(model)
        @test params == (τ = Real,)
    end

    @testset "Parameter Validation" begin
        W = sparse(Bool[0 1; 1 0])
        model = BesagModel(W)

        @test precision_matrix(model; τ = 1.0) isa AbstractMatrix
        @test_throws ArgumentError precision_matrix(model; τ = 0.0)
        @test_throws ArgumentError precision_matrix(model; τ = -1.0)
    end

    @testset "Precision Matrix Structure" begin
        # Triangle graph: each node connected to other 2
        W = sparse(Bool[0 1 1; 1 0 1; 1 1 0])
        model = BesagModel(W)
        τ = 2.0
        Q = precision_matrix(model; τ = τ)

        # Check Laplacian structure: Q = τ*(D-W) + regularization*I
        degrees = vec(sum(W, dims = 2))  # [2, 2, 2]
        D = Diagonal(degrees)
        expected = τ * (D - W) + model.regularization * I
        @test Matrix(Q) ≈ Matrix(expected)
    end

    @testset "Mean and Constraints" begin
        W = sparse(Bool[0 1 1; 1 0 1; 1 1 0])
        model = BesagModel(W)

        @test mean(model; τ = 1.0) == zeros(3)

        constraint_info = constraints(model; τ = 1.0)
        @test constraint_info !== nothing
        A, e = constraint_info
        @test A == ones(1, 3)  # Sum-to-zero constraint
        @test e == [0.0]
    end

    @testset "ConstrainedGMRF Construction" begin
        W = sparse(Bool[0 1 1; 1 0 1; 1 1 0])
        model = BesagModel(W)
        τ = 1.2
        gmrf = model(τ = τ)

        @test gmrf isa ConstrainedGMRF  # Should be constrained due to sum-to-zero
        @test length(gmrf) == 3
        @test gmrf.constraint_matrix == ones(1, 3)
        @test gmrf.constraint_vector == [0.0]
    end

    @testset "Type Stability" begin
        W = sparse(Bool[0 1; 1 0])
        model = BesagModel(W)

        Q = precision_matrix(model; τ = 1.0)
        @test eltype(Q) == Float64

        gmrf = model(τ = 1.0)
        @test gmrf isa ConstrainedGMRF{Float64}
    end
end
