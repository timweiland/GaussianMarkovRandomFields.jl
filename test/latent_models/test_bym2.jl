using GaussianMarkovRandomFields
using LinearAlgebra
using SparseArrays
using LinearSolve

@testset "BYM2Model" begin
    @testset "Constructor" begin
        # Valid triangle graph
        W = sparse(Bool[0 1 1; 1 0 1; 1 1 0])
        model = BYM2Model(W)
        @test model.n == 3
        @test length(model) == 6  # 2n dimensional (spatial + unstructured)
        @test model.besag isa BesagModel
        @test model.iid isa IIDModel

        # Matrix input should be converted to sparse by BesagModel
        W_dense = Bool[0 1; 1 0]
        model_dense = BYM2Model(W_dense)
        @test model_dense.besag.adjacency isa SparseMatrixCSC

        # BYM2 requires variance normalization
        @test_throws ArgumentError BYM2Model(W; normalize_var = Val(false))

        # Invalid adjacency matrices should be caught by BesagModel
        @test_throws ArgumentError BYM2Model([1 0; 0 1; 1 0])  # Not square
        @test_throws ArgumentError BYM2Model([0 1; 0 0])  # Not symmetric
        @test_throws ArgumentError BYM2Model([1 1; 1 0])  # Non-zero diagonal
    end

    @testset "Hyperparameters" begin
        W = sparse(Bool[0 1; 1 0])
        model = BYM2Model(W)
        params = hyperparameters(model)
        @test params == (τ = Real, φ = Real)
    end

    @testset "Parameter Validation" begin
        W = sparse(Bool[0 1; 1 0])
        model = BYM2Model(W)

        # Valid parameters
        @test precision_matrix(model; τ = 1.0, φ = 0.5) isa AbstractMatrix

        # Invalid τ
        @test_throws ArgumentError precision_matrix(model; τ = 0.0, φ = 0.5)
        @test_throws ArgumentError precision_matrix(model; τ = -1.0, φ = 0.5)

        # Invalid φ (must be in (0, 1))
        @test_throws ArgumentError precision_matrix(model; τ = 1.0, φ = 0.0)
        @test_throws ArgumentError precision_matrix(model; τ = 1.0, φ = 1.0)
        @test_throws ArgumentError precision_matrix(model; τ = 1.0, φ = -0.1)
        @test_throws ArgumentError precision_matrix(model; τ = 1.0, φ = 1.1)
    end

    @testset "Precision Matrix Structure" begin
        # Triangle graph
        W = sparse(Bool[0 1 1; 1 0 1; 1 1 0])
        model = BYM2Model(W)
        τ = 2.0
        φ = 0.5
        Q = precision_matrix(model; τ = τ, φ = φ)

        # Check dimensions
        @test size(Q) == (6, 6)

        # Check block-diagonal structure (off-diagonal blocks should be zero)
        @test all(Q[1:3, 4:6] .== 0)
        @test all(Q[4:6, 1:3] .== 0)

        # Extract blocks
        Q_spatial = Q[1:3, 1:3]
        Q_unstructured = Q[4:6, 4:6]

        # Unstructured block should be diagonal with τ/φ on diagonal
        expected_unstructured = (τ / φ) * I(3)
        @test Matrix(Q_unstructured) ≈ Matrix(expected_unstructured)

        # Spatial block should be τ/(1-φ) times the scaled Besag precision
        # (We can't easily verify the exact structure due to normalization,
        # but we can check it's the right scaling)
        Q_besag_unit = precision_matrix(model.besag; τ = 1.0)
        expected_spatial = (τ / (1 - φ)) * Q_besag_unit
        @test Matrix(Q_spatial) ≈ Matrix(expected_spatial)
    end

    @testset "Variance Properties" begin
        # Simple 2-node chain
        W = sparse([0 1; 1 0])
        model = BYM2Model(W)
        τ = 1.0
        φ = 0.5

        gmrf = model(τ = τ, φ = φ)
        v = var(gmrf)

        # Components 1:2 are spatial, 3:4 are unstructured
        v_spatial = v[1:2]
        v_unstructured = v[3:4]

        # Unstructured variances should be close to φ/τ = 0.5
        @test all(isapprox.(v_unstructured, fill(φ / τ, 2); rtol = 0.1))

        # Spatial variances should have geometric mean close to (1-φ)/τ = 0.5
        _geomean = x -> exp(mean(log.(x)))
        @test isapprox(_geomean(v_spatial), (1 - φ) / τ; rtol = 0.2)
    end

    @testset "Mixing Parameter Effect" begin
        W = sparse([0 1 1; 1 0 1; 1 1 0])
        model = BYM2Model(W)
        τ = 1.0

        # φ ≈ 0: mostly spatial
        φ_spatial = 0.1
        gmrf_spatial = model(τ = τ, φ = φ_spatial)
        v_spatial = var(gmrf_spatial)

        # φ ≈ 1: mostly unstructured
        φ_unstructured = 0.9
        gmrf_unstructured = model(τ = τ, φ = φ_unstructured)
        v_unstructured = var(gmrf_unstructured)

        # When φ is small, unstructured component (4:6) should have smaller variance
        @test mean(v_spatial[4:6]) < mean(v_spatial[1:3])

        # When φ is large, unstructured component (4:6) should have larger variance
        @test mean(v_unstructured[4:6]) > mean(v_unstructured[1:3])
    end

    @testset "Mean and Constraints" begin
        W = sparse(Bool[0 1 1; 1 0 1; 1 1 0])
        model = BYM2Model(W)

        # Mean should be zero for all 2n components
        @test mean(model; τ = 1.0, φ = 0.5) == zeros(6)

        # Constraints should be sum-to-zero on spatial component only
        constraint_info = constraints(model; τ = 1.0, φ = 0.5)
        @test constraint_info !== nothing
        A, e = constraint_info
        @test size(A) == (1, 6)
        # First 3 components (spatial) should sum to zero
        @test A[1, 1:3] ≈ ones(3)
        # Last 3 components (unstructured) are unconstrained
        @test all(A[1, 4:6] .== 0)
        @test e == [0.0]
    end

    @testset "ConstrainedGMRF Construction" begin
        W = sparse(Bool[0 1 1; 1 0 1; 1 1 0])
        model = BYM2Model(W)
        τ = 1.2
        φ = 0.4
        gmrf = model(τ = τ, φ = φ)

        @test gmrf isa ConstrainedGMRF  # Should be constrained due to sum-to-zero
        @test length(gmrf) == 6  # 2n dimensional
        @test size(gmrf.constraint_matrix) == (1, 6)
        @test gmrf.constraint_vector == [0.0]
    end

    @testset "Type Stability" begin
        W = sparse(Bool[0 1; 1 0])
        model = BYM2Model(W)

        Q = precision_matrix(model; τ = 1.0, φ = 0.5)
        @test eltype(Q) == Float64

        gmrf = model(τ = 1.0, φ = 0.5)
        @test gmrf isa ConstrainedGMRF{Float64}
    end

    @testset "Model Name" begin
        W = sparse(Bool[0 1; 1 0])
        model = BYM2Model(W)
        @test model_name(model) == :bym2
    end

    @testset "Algorithm Storage and Passing" begin
        W = sparse([0 1 0; 1 0 1; 0 1 0])

        # Test default algorithm
        model = BYM2Model(W)
        @test model.alg isa CHOLMODFactorization

        # Test algorithm is passed to GMRF
        constrained_gmrf = model(τ = 1.0, φ = 0.5)
        @test constrained_gmrf.base_gmrf.linsolve_cache.alg isa CHOLMODFactorization

        # Test custom algorithm
        custom_model = BYM2Model(W, alg = LDLtFactorization())
        @test custom_model.alg isa LDLtFactorization
    end

    @testset "Singleton Policy" begin
        # Single-node graph
        W1 = spzeros(1, 1)
        τ = 2.0
        φ = 0.5

        # Gaussian policy
        m_g = BYM2Model(W1; singleton_policy = Val(:gaussian))
        x_g = m_g(τ = τ, φ = φ)
        @test length(x_g) == 2  # 2n = 2

        # Degenerate policy
        m_d = BYM2Model(W1; singleton_policy = Val(:degenerate))
        x_d = m_d(τ = τ, φ = φ)
        @test length(x_d) == 2
    end

    @testset "Comparison with Classic BYM" begin
        # BYM2 with φ=0.5 and τ=1 should give similar results to
        # classic BYM with τ_spatial=2 and τ_iid=2
        W = sparse([0 1 1; 1 0 1; 1 1 0])

        # BYM2
        bym2 = BYM2Model(W)
        gmrf_bym2 = bym2(τ = 1.0, φ = 0.5)
        v_bym2 = var(gmrf_bym2)

        # Classic BYM using CombinedModel
        besag = BesagModel(W; normalize_var = Val(true))
        iid = IIDModel(3)
        bym_classic = CombinedModel([besag, iid])
        gmrf_classic = bym_classic(τ_besag = 2.0, τ_iid = 2.0)
        v_classic = var(gmrf_classic)

        # Total variance should be similar (sum of spatial and unstructured)
        # In BYM2, effects are added: η = u* + v*
        # In classic BYM, components are separate
        # So we can't directly compare individual variances,
        # but the structure should be consistent
        @test length(v_bym2) == 6
        @test length(v_classic) == 6
    end

    @testset "IID Component Constraints (identifiability)" begin
        W = sparse([0 1; 1 0])

        # Default: Only Besag has constraints, IID doesn't
        model_unconstrained = BYM2Model(W)
        A, e = constraints(model_unconstrained; τ = 1.0, φ = 0.5)
        @test size(A, 1) == 1  # Only Besag sum-to-zero constraint
        @test size(A, 2) == 4  # 2n dimensional
        # First 2 components (spatial) should sum to zero
        @test A[1, 1:2] ≈ ones(2)
        @test all(A[1, 3:4] .== 0)

        # With IID constraint: Both components should be constrained
        model_constrained = BYM2Model(W; iid_constraint = :sumtozero)
        A2, e2 = constraints(model_constrained; τ = 1.0, φ = 0.5)
        @test size(A2, 1) == 2  # Two sum-to-zero constraints
        @test size(A2, 2) == 4  # 2n dimensional

        # First constraint: spatial component sums to zero
        @test A2[1, 1:2] ≈ ones(2)
        @test all(A2[1, 3:4] .== 0)

        # Second constraint: IID component sums to zero
        @test all(A2[2, 1:2] .== 0)
        @test A2[2, 3:4] ≈ ones(2)

        # Verify GMRFs are created correctly
        gmrf_unconstrained = model_unconstrained(τ = 1.0, φ = 0.5)
        gmrf_constrained = model_constrained(τ = 1.0, φ = 0.5)

        @test gmrf_unconstrained isa ConstrainedGMRF
        @test gmrf_constrained isa ConstrainedGMRF
        @test size(gmrf_unconstrained.constraint_matrix, 1) == 1
        @test size(gmrf_constrained.constraint_matrix, 1) == 2
    end
end
