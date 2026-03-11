using GaussianMarkovRandomFields
using LinearAlgebra
using SparseArrays
using LinearSolve

# Helper: build k-th order difference operator for verifying precision matrices
function _test_diff_operator(n, order)
    D = sparse(1.0I, n, n)  # start with identity
    for _ in 1:order
        m = size(D, 1)
        D1 = spzeros(m - 1, n)
        for i in 1:(m - 1)
            D1[i, :] = D[i + 1, :] - D[i, :]
        end
        D = D1
    end
    return D
end

@testset "RWModel" begin
    # ============================================================
    # RW1 backward compatibility (RW1Model = RWModel{1})
    # ============================================================
    @testset "RW1 Backward Compatibility" begin
        @testset "Type alias" begin
            @test RW1Model === RWModel{1}
        end

        @testset "Constructor" begin
            model = RW1Model(5)
            @test model.n == 5
            @test model.regularization == 1.0e-5
            @test model isa RWModel{1}

            model_custom = RW1Model(3; regularization = 1.0e-4)
            @test model_custom.regularization == 1.0e-4

            @test_throws ArgumentError RW1Model(1)  # n must be > 1
            @test_throws ArgumentError RW1Model(0)
            @test_throws ArgumentError RW1Model(5; regularization = -1.0)
        end

        @testset "Hyperparameters" begin
            model = RW1Model(3)
            params = hyperparameters(model)
            @test params == (τ = Real,)
        end

        @testset "Parameter Validation" begin
            model = RW1Model(3)
            @test precision_matrix(model; τ = 1.0) isa SymTridiagonal
            @test precision_matrix(model; τ = 2.0) isa SymTridiagonal
            @test_throws ArgumentError precision_matrix(model; τ = 0.0)
            @test_throws ArgumentError precision_matrix(model; τ = -1.0)
        end

        @testset "Precision Matrix Structure" begin
            @testset "n=3 case" begin
                model = RW1Model(3)
                τ = 1.5
                Q = precision_matrix(model; τ = τ)
                @test Q isa SymTridiagonal
                @test size(Q) == (3, 3)

                expected = τ * [
                    1  -1   0;
                    -1  2  -1;
                    0  -1   1
                ] + model.regularization * I(3)
                @test Matrix(Q) ≈ expected
            end

            @testset "Regularization effect" begin
                model = RW1Model(3; regularization = 1.0e-3)
                τ = 1.0
                Q = precision_matrix(model; τ = τ)
                @test Q[1, 1] == τ + 1.0e-3
                @test Q[2, 2] == 2τ + 1.0e-3
                @test Q[3, 3] == τ + 1.0e-3
                @test Q[1, 2] == -τ
            end

            @testset "Scaling by τ" begin
                model = RW1Model(4)
                τ = 2.0
                Q = precision_matrix(model; τ = τ)
                @test Q[1, 1] == τ + model.regularization
                @test Q[4, 4] == τ + model.regularization
                @test Q[2, 2] == 2τ + model.regularization
                @test Q[3, 3] == 2τ + model.regularization
                @test Q[1, 2] == -τ
                @test Q[2, 3] == -τ
                @test Q[3, 4] == -τ
            end
        end

        @testset "Mean Vector" begin
            model = RW1Model(5)
            μ = mean(model; τ = 1.0)
            @test μ == zeros(5)
            @test length(μ) == 5
        end

        @testset "Constraints" begin
            model = RW1Model(5)
            constraint_info = constraints(model; τ = 1.0)
            @test constraint_info !== nothing
            A, e = constraint_info
            @test size(A) == (1, 5)
            @test A == ones(1, 5)
            @test e == [0.0]
        end

        @testset "ConstrainedGMRF Construction" begin
            model = RW1Model(4)
            τ = 1.2
            gmrf = model(τ = τ)
            @test gmrf isa ConstrainedGMRF
            @test length(gmrf) == 4
            @test mean(gmrf) == zeros(4)
            @test gmrf.constraint_matrix == ones(1, 4)
            @test gmrf.constraint_vector == [0.0]
        end

        @testset "Type Stability" begin
            model = RW1Model(3)
            Q_f64 = precision_matrix(model; τ = 1.0)
            @test eltype(Q_f64) == Float64
            gmrf = model(τ = 1.0)
            @test gmrf isa ConstrainedGMRF{Float64}
        end

        @testset "Algorithm Storage and Passing" begin
            model = RW1Model(10)
            @test model.alg isa LDLtFactorization
            constrained_gmrf = model(τ = 1.0)
            @test constrained_gmrf.base_gmrf.linsolve_cache.alg isa LDLtFactorization

            custom_model = RW1Model(10, alg = CHOLMODFactorization())
            @test custom_model.alg isa CHOLMODFactorization
            custom_constrained = custom_model(τ = 1.0)
            @test custom_constrained.base_gmrf.linsolve_cache.alg isa CHOLMODFactorization
        end

        @testset "Additional Constraints" begin
            A_add = [1.0 0.0 0.0 0.0 0.0]
            e_add = [0.0]
            model = RW1Model(5, additional_constraints = (A_add, e_add))
            A, e = constraints(model)
            @test size(A, 1) == 2
            @test A[1, :] ≈ ones(5)
            @test e[1] ≈ 0.0
            @test A[2, :] ≈ vec(A_add)
            @test e[2] ≈ e_add[1]
            gmrf = model(τ = 1.0)
            @test gmrf isa ConstrainedGMRF
        end

        @testset "Model name" begin
            model = RW1Model(5)
            @test model_name(model) == :rw1
        end
    end

    # ============================================================
    # RW2Model (RWModel{2})
    # ============================================================
    @testset "RW2Model" begin
        @testset "Type alias" begin
            @test RW2Model === RWModel{2}
        end

        @testset "Constructor" begin
            model = RW2Model(5)
            @test model.n == 5
            @test model.regularization == 1.0e-5
            @test model isa RWModel{2}

            # n must be > 2 for RW2
            @test_throws ArgumentError RW2Model(2)
            @test_throws ArgumentError RW2Model(1)
            @test_throws ArgumentError RW2Model(0)
            @test_throws ArgumentError RW2Model(5; regularization = -1.0)
        end

        @testset "Hyperparameters" begin
            model = RW2Model(5)
            params = hyperparameters(model)
            @test params == (τ = Real,)
        end

        @testset "Parameter Validation" begin
            model = RW2Model(5)
            @test_throws ArgumentError precision_matrix(model; τ = 0.0)
            @test_throws ArgumentError precision_matrix(model; τ = -1.0)
        end

        @testset "Default solver" begin
            model = RW2Model(5)
            @test model.alg isa CHOLMODFactorization
        end

        @testset "Precision Matrix Structure" begin
            @testset "Returns sparse matrix" begin
                model = RW2Model(5)
                Q = precision_matrix(model; τ = 1.0)
                @test Q isa SparseMatrixCSC
            end

            @testset "n=5, τ=1 correctness" begin
                model = RW2Model(5; regularization = 0.0)
                τ = 1.0
                Q = precision_matrix(model; τ = τ)

                # D2 for n=5 is 3x5: [1 -2 1 0 0; 0 1 -2 1 0; 0 0 1 -2 1]
                # Q = D2'*D2
                D2 = _test_diff_operator(5, 2)
                expected = τ * Matrix(D2' * D2)
                @test Matrix(Q) ≈ expected
            end

            @testset "n=5, τ=1 known values" begin
                # Verify against the hard-coded values from the INLA formula interface
                model = RW2Model(5; regularization = 0.0)
                Q = precision_matrix(model; τ = 1.0)
                Qm = Matrix(Q)

                # Known RW2 precision for n=5:
                # [1  -2   1   0   0]
                # [-2  5  -4   1   0]
                # [1  -4   6  -4   1]
                # [0   1  -4   5  -2]
                # [0   0   1  -2   1]
                expected = [
                    1  -2   1   0   0;
                    -2   5  -4   1   0;
                    1  -4   6  -4   1;
                    0   1  -4   5  -2;
                    0   0   1  -2   1
                ]
                @test Qm ≈ Float64.(expected)
            end

            @testset "Scaling by τ" begin
                model = RW2Model(5; regularization = 0.0)
                τ = 2.5
                Q = precision_matrix(model; τ = τ)
                Q1 = precision_matrix(model; τ = 1.0)
                @test Matrix(Q) ≈ τ * Matrix(Q1)
            end

            @testset "Regularization effect" begin
                reg = 1.0e-3
                model = RW2Model(5; regularization = reg)
                τ = 1.0
                Q = precision_matrix(model; τ = τ)
                model_noreg = RW2Model(5; regularization = 0.0)
                Q_noreg = precision_matrix(model_noreg; τ = τ)
                @test Matrix(Q) ≈ Matrix(Q_noreg) + reg * I(5)
            end

            @testset "Symmetry" begin
                model = RW2Model(10)
                Q = precision_matrix(model; τ = 1.5)
                @test issymmetric(Matrix(Q))
            end

            @testset "Positive semidefinite (rank n-2)" begin
                model = RW2Model(10; regularization = 0.0)
                Q = precision_matrix(model; τ = 1.0)
                evals = eigvals(Symmetric(Matrix(Q)))
                # Should have exactly 2 zero eigenvalues
                @test count(e -> e < 1.0e-10, evals) == 2
                # Remaining eigenvalues should be positive
                @test all(e -> e > -1.0e-10, evals)
            end
        end

        @testset "Mean Vector" begin
            model = RW2Model(5)
            μ = mean(model; τ = 1.0)
            @test μ == zeros(5)
        end

        @testset "Constraints" begin
            model = RW2Model(5)
            A, e = constraints(model; τ = 1.0)
            # RW2 needs 2 constraints (constant + linear null space)
            @test size(A, 1) == 2
            @test size(A, 2) == 5
            # First constraint: sum-to-zero (constant polynomial)
            @test A[1, :] ≈ ones(5)
            # Second constraint: linear polynomial sum-to-zero
            @test A[2, :] ≈ [1.0, 2.0, 3.0, 4.0, 5.0]
            @test e == zeros(2)
        end

        @testset "ConstrainedGMRF Construction" begin
            model = RW2Model(5)
            gmrf = model(τ = 1.0)
            @test gmrf isa ConstrainedGMRF
            @test length(gmrf) == 5
            @test mean(gmrf) == zeros(5)
        end

        @testset "Model name" begin
            model = RW2Model(5)
            @test model_name(model) == :rw2
        end

        @testset "Additional Constraints" begin
            A_add = [1.0 0.0 0.0 0.0 0.0]
            e_add = [0.0]
            model = RW2Model(5; additional_constraints = (A_add, e_add))
            A, e = constraints(model)
            @test size(A, 1) == 3  # 2 intrinsic + 1 additional
            gmrf = model(τ = 1.0)
            @test gmrf isa ConstrainedGMRF
        end
    end

    # ============================================================
    # RW3 (RWModel{3}) — basic tests
    # ============================================================
    @testset "RWModel{3}" begin
        @testset "Constructor" begin
            model = RWModel{3}(10)
            @test model.n == 10
            @test model isa RWModel{3}
            # n must be > 3
            @test_throws ArgumentError RWModel{3}(3)
            @test_throws ArgumentError RWModel{3}(2)
        end

        @testset "Precision matrix correctness" begin
            model = RWModel{3}(8; regularization = 0.0)
            Q = precision_matrix(model; τ = 1.0)
            D3 = _test_diff_operator(8, 3)
            expected = Matrix(D3' * D3)
            @test Matrix(Q) ≈ expected
        end

        @testset "Rank n-3" begin
            model = RWModel{3}(10; regularization = 0.0)
            Q = precision_matrix(model; τ = 1.0)
            evals = eigvals(Symmetric(Matrix(Q)))
            @test count(e -> e < 1.0e-10, evals) == 3
            @test all(e -> e > -1.0e-10, evals)
        end

        @testset "Constraints" begin
            model = RWModel{3}(6)
            A, e = constraints(model; τ = 1.0)
            @test size(A, 1) == 3  # 3 constraints for RW3
            @test size(A, 2) == 6
            @test A[1, :] ≈ ones(6)                             # constant
            @test A[2, :] ≈ [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]    # linear
            @test A[3, :] ≈ [1.0, 4.0, 9.0, 16.0, 25.0, 36.0] # quadratic
            @test e == zeros(3)
        end

        @testset "Model name" begin
            model = RWModel{3}(10)
            @test model_name(model) == :rw3
        end

        @testset "ConstrainedGMRF Construction" begin
            model = RWModel{3}(10)
            gmrf = model(τ = 1.0)
            @test gmrf isa ConstrainedGMRF
            @test length(gmrf) == 10
        end
    end
end
