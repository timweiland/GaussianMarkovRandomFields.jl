using GaussianMarkovRandomFields
using LinearAlgebra
using SparseArrays

@testset "SeparableModel" begin
    @testset "Constructor" begin
        rw1 = RW1Model(5)
        iid = IIDModel(3)
        besag_W = sparse([0 1 0; 1 0 1; 0 1 0])
        besag = BesagModel(besag_W)

        # Valid 2-component model
        sep = SeparableModel(rw1, iid)
        @test sep.components == (rw1, iid)
        @test sep.alg === nothing

        # Valid 3-component model
        sep3 = SeparableModel(rw1, iid, besag)
        @test length(sep3.components) == 3

        # Invalid: single component
        @test_throws ErrorException SeparableModel(rw1)

        # Custom algorithm
        sep_alg = SeparableModel(rw1, iid; alg = :cg)
        @test sep_alg.alg == :cg
    end

    @testset "Length" begin
        rw1 = RW1Model(10)
        iid = IIDModel(5)
        sep = SeparableModel(rw1, iid)
        @test length(sep) == 10 * 5  # 50

        # 3-way product
        ar1 = AR1Model(4)
        sep3 = SeparableModel(rw1, iid, ar1)
        @test length(sep3) == 10 * 5 * 4  # 200
    end

    @testset "Model Name" begin
        rw1 = RW1Model(5)
        iid = IIDModel(3)
        sep = SeparableModel(rw1, iid)
        @test model_name(sep) == :separable
    end

    @testset "Hyperparameters" begin
        rw1 = RW1Model(5)
        iid = IIDModel(3)
        sep = SeparableModel(rw1, iid)
        params = hyperparameters(sep)

        # Should have τ_rw1 and τ_iid
        @test haskey(params, :τ_rw1)
        @test haskey(params, :τ_iid)
        @test params[:τ_rw1] == Real
        @test params[:τ_iid] == Real
    end

    @testset "Hyperparameters with duplicate models" begin
        rw1_a = RW1Model(5)
        rw1_b = RW1Model(3)
        sep = SeparableModel(rw1_a, rw1_b)
        params = hyperparameters(sep)

        # Should have τ_rw1 and τ_rw1_2 for disambiguation
        @test haskey(params, :τ_rw1)
        @test haskey(params, :τ_rw1_2)
    end

    @testset "Mean Computation" begin
        rw1 = RW1Model(3)
        iid = IIDModel(2)
        sep = SeparableModel(rw1, iid)

        # Both models have zero mean, so result should be zero
        m = mean(sep; τ_rw1 = 1.0, τ_iid = 1.0)
        @test m ≈ zeros(6)
        @test length(m) == 6
    end

    @testset "Precision Matrix Structure" begin
        @testset "Q = Q_1 ⊗ Q_2 ordering" begin
            # Simple 3×2 case
            rw1 = RW1Model(3)
            iid = IIDModel(2)
            sep = SeparableModel(rw1, iid)

            Q = precision_matrix(sep; τ_rw1 = 1.0, τ_iid = 1.0)
            Q_dense = Matrix(Q)

            # Get component matrices for manual verification
            Q_rw1 = Matrix(precision_matrix(rw1; τ = 1.0))
            Q_iid = Matrix(precision_matrix(iid; τ = 1.0))

            # Expected: Q_rw1 ⊗ Q_iid
            Q_expected = kron(Q_rw1, Q_iid)

            @test Q_dense ≈ Q_expected
        end

        @testset "Block-tridiagonal structure (time ⊗ space)" begin
            # RW1 (tridiagonal) ⊗ IID should give block-tridiagonal
            rw1 = RW1Model(3)
            iid = IIDModel(2)
            sep = SeparableModel(rw1, iid)

            Q = Matrix(precision_map(sep(τ_rw1 = 1.0, τ_iid = 1.0)))

            # Check block structure: 3 blocks of 2×2, with tridiagonal block pattern
            # Q[1:2, 1:2] should be nonzero (main diagonal block)
            # Q[1:2, 3:4] should be nonzero (first off-diagonal block)
            # Q[1:2, 5:6] should be zero (too far away)
            @test norm(Q[1:2, 1:2]) > 0
            @test norm(Q[1:2, 3:4]) > 0
            @test Q[1:2, 5:6] ≈ zeros(2, 2)
            @test Q[5:6, 1:2] ≈ zeros(2, 2)
        end
    end

    @testset "Constraint Composition" begin
        @testset "Single constraint (time)" begin
            # RW1 has sum-to-zero constraint, IID has none
            rw1 = RW1Model(3)
            iid = IIDModel(2)
            sep = SeparableModel(rw1, iid)

            constr = constraints(sep; τ_rw1 = 1.0, τ_iid = 1.0)
            @test constr !== nothing
            A, e = constr

            # Should have 2 constraints (one for each spatial location)
            # Pattern: [1 0 1 0 1 0; 0 1 0 1 0 1] (A_rw1 ⊗ I_2)
            @test size(A, 1) == 2
            @test size(A, 2) == 6
            @test vec(e) ≈ zeros(2)

            # Verify structure matches A_rw1 ⊗ I_iid
            A_rw1 = [1 1 1]
            I_2 = Matrix(I, 2, 2)
            A_expected = kron(A_rw1, I_2)
            @test Matrix(A) ≈ A_expected
        end

        @testset "Constraint redundancy removal" begin
            # Both RW1 components have sum-to-zero → redundancy
            rw1_a = RW1Model(3)
            rw1_b = RW1Model(2)
            sep = SeparableModel(rw1_a, rw1_b)

            constr = constraints(sep; τ_rw1 = 1.0, τ_rw1_2 = 1.0)
            @test constr !== nothing
            A, e = constr

            # Without redundancy removal: 2 + 3 = 5 constraints
            # With redundancy removal: should have 4 (one redundant)
            @test size(A, 1) <= 4
            @test rank(Matrix(A)) == size(A, 1)  # Full row rank
        end

        @testset "Correct Kronecker ordering for constraints" begin
            # Verify I_before ⊗ A_i ⊗ I_after pattern
            rw1 = RW1Model(2)  # Has sum-to-zero
            iid = IIDModel(3)  # No constraint
            sep = SeparableModel(rw1, iid)

            constr = constraints(sep; τ_rw1 = 1.0, τ_iid = 1.0)
            A, e = constr

            # Component 1 (RW1): n_before=1, n_after=3
            # A_full = I_1 ⊗ A_rw1 ⊗ I_3 = A_rw1 ⊗ I_3
            A_rw1 = [1 1]
            I_3 = Matrix(I, 3, 3)
            A_expected = kron(A_rw1, I_3)

            @test Matrix(A) ≈ A_expected
        end
    end

    @testset "GMRF Instantiation" begin
        @testset "Basic instantiation" begin
            rw1 = RW1Model(5)
            iid = IIDModel(3)
            sep = SeparableModel(rw1, iid)

            gmrf = sep(τ_rw1 = 1.0, τ_iid = 1.0)
            @test length(gmrf) == 15
            @test mean(gmrf) ≈ zeros(15)
        end

        @testset "With constraints" begin
            rw1 = RW1Model(4)
            iid = IIDModel(2)
            sep = SeparableModel(rw1, iid)

            gmrf = sep(τ_rw1 = 1.0, τ_iid = 1.0)
            # Should be a ConstrainedGMRF (or GMRF)
            @test length(gmrf) == 8
        end

        @testset "Different hyperparameter values" begin
            rw1 = RW1Model(3)
            iid = IIDModel(2)
            sep = SeparableModel(rw1, iid)

            gmrf1 = sep(τ_rw1 = 1.0, τ_iid = 2.0)
            gmrf2 = sep(τ_rw1 = 2.0, τ_iid = 1.0)

            Q1 = Matrix(precision_map(gmrf1))
            Q2 = Matrix(precision_map(gmrf2))

            # Precision matrices should be different (scaled differently)
            @test !(Q1 ≈ Q2)
        end
    end

    @testset "3-way Separable Models" begin
        rw1 = RW1Model(3)
        iid = IIDModel(2)
        ar1 = AR1Model(4)
        sep = SeparableModel(rw1, iid, ar1)

        @test length(sep) == 3 * 2 * 4  # 24

        # Test precision matrix
        Q = precision_matrix(sep; τ_rw1 = 1.0, τ_iid = 1.0, τ_ar1 = 1.0, ρ_ar1 = 0.8)
        @test size(Q) == (24, 24)

        # Test GMRF creation
        gmrf = sep(τ_rw1 = 1.0, τ_iid = 1.0, τ_ar1 = 1.0, ρ_ar1 = 0.8)
        @test length(gmrf) == 24
    end

    @testset "Edge cases" begin
        @testset "Minimal sizes" begin
            rw1 = RW1Model(2)  # Minimum for RW1
            iid = IIDModel(1)  # Minimum for IID
            sep = SeparableModel(rw1, iid)

            @test length(sep) == 2
            gmrf = sep(τ_rw1 = 1.0, τ_iid = 1.0)
            @test length(gmrf) == 2
        end

        @testset "Large component counts" begin
            # Many components (5-way)
            comps = [IIDModel(2) for _ in 1:5]
            sep = SeparableModel(comps...)

            @test length(sep) == 2^5  # 32
            params = hyperparameters(sep)
            @test length(keys(params)) == 5
        end
    end
end
