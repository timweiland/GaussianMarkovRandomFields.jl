using GaussianMarkovRandomFields
using LinearAlgebra
using SparseArrays
using LinearSolve

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
        # Isolated nodes are allowed; handled per singleton policy
        @test BesagModel([0 0; 0 0]) isa BesagModel
        @test_throws ArgumentError BesagModel(W; regularization = 0.0)  # Bad regularization
    end

    @testset "Hyperparameters" begin
        W = sparse(Bool[0 1; 1 0])
        model = BesagModel(W; normalize_var = Val(false))
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
        model = BesagModel(W; normalize_var = Val(false))
        τ = 2.0
        Q = precision_matrix(model; τ = τ)

        # Check Laplacian structure: Q = τ*(D-W) + regularization*I
        degrees = vec(sum(W, dims = 2))  # [2, 2, 2]
        D = Diagonal(degrees)
        expected = τ * (D - W) + model.regularization * I
        @test Matrix(Q) ≈ Matrix(expected)
    end

    @testset "Normalization scales per component (normalize_var=true)" begin
        # Two components: 3-node chain and 2-node chain
        W = spzeros(5, 5)
        W[1, 2] = 1; W[2, 1] = 1
        W[2, 3] = 1; W[3, 2] = 1
        W[4, 5] = 1; W[5, 4] = 1

        model = BesagModel(W; normalize_var = Val(true))
        x = model(τ = 1.0)
        v = var(x)

        _geomean = x -> exp(mean(log.(x)))
        g1 = _geomean(v[[1, 2, 3]])
        g2 = _geomean(v[[4, 5]])
        @test isapprox(g1, 1.0; atol = 0.1, rtol = 0.1)
        @test isapprox(g2, 1.0; atol = 0.1, rtol = 0.1)
    end

    @testset "Mean and Constraints" begin
        W = sparse(Bool[0 1 1; 1 0 1; 1 1 0])
        model = BesagModel(W; normalize_var = Val(false))

        @test mean(model; τ = 1.0) == zeros(3)

        constraint_info = constraints(model; τ = 1.0)
        @test constraint_info !== nothing
        A, e = constraint_info
        @test A == ones(1, 3)  # Sum-to-zero constraint
        @test e == [0.0]
    end

    @testset "ConstrainedGMRF Construction" begin
        W = sparse(Bool[0 1 1; 1 0 1; 1 1 0])
        model = BesagModel(W; normalize_var = Val(false))
        τ = 1.2
        gmrf = model(τ = τ)

        @test gmrf isa ConstrainedGMRF  # Should be constrained due to sum-to-zero
        @test length(gmrf) == 3
        @test gmrf.constraint_matrix == ones(1, 3)
        @test gmrf.constraint_vector == [0.0]
    end

    @testset "Type Stability" begin
        W = sparse(Bool[0 1; 1 0])
        model = BesagModel(W; normalize_var = Val(false))

        Q = precision_matrix(model; τ = 1.0)
        @test eltype(Q) == Float64

        gmrf = model(τ = 1.0)
        @test gmrf isa ConstrainedGMRF{Float64}
    end

    @testset "Singleton policy: gaussian vs degenerate" begin
        # Single-node graph
        W1 = spzeros(1, 1)
        τ = 2.5

        # Gaussian policy: proper variance ~ 1/(τ + reg)
        m_g = BesagModel(W1; normalize_var = Val(false), singleton_policy = Val(:gaussian))
        x_g = m_g(τ = τ)
        v_g = var(x_g)
        @test isapprox(v_g[1], 1.0 / (τ + m_g.regularization); atol = 1.0e-6, rtol = 1.0e-3)

        # Degenerate policy: constrained to zero ⇒ zero variance
        m_d = BesagModel(W1; normalize_var = Val(false), singleton_policy = Val(:degenerate))
        x_d = m_d(τ = τ)
        v_d = var(x_d)
        @test isapprox(v_d[1], 0.0; atol = 1.0e-10)

        # Two singletons
        W2 = spzeros(2, 2)
        m_g2 = BesagModel(W2; normalize_var = Val(false), singleton_policy = Val(:gaussian))
        x_g2 = m_g2(τ = τ)
        v_g2 = var(x_g2)
        @test all(isapprox.(v_g2, fill(1.0 / (τ + m_g2.regularization), 2); atol = 1.0e-6, rtol = 1.0e-3))

        m_d2 = BesagModel(W2; normalize_var = Val(false), singleton_policy = Val(:degenerate))
        x_d2 = m_d2(τ = τ)
        v_d2 = var(x_d2)
        @test all(isapprox.(v_d2, zeros(2); atol = 1.0e-10))
    end

    @testset "Algorithm Storage and Passing" begin
        # Create simple adjacency matrix
        W = sparse([0 1 0; 1 0 1; 0 1 0])

        # Test default algorithm (CHOLMODFactorization for sparse general)
        model = BesagModel(W)
        @test model.alg isa CHOLMODFactorization

        # Test algorithm is passed to GMRF (Besag creates ConstrainedGMRF)
        constrained_gmrf = model(τ = 1.0)
        @test constrained_gmrf.base_gmrf.linsolve_cache.alg isa CHOLMODFactorization

        # Test custom algorithm
        custom_model = BesagModel(W, alg = LDLtFactorization())
        @test custom_model.alg isa LDLtFactorization
    end
end
