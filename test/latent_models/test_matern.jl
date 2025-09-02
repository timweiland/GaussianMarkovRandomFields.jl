using GaussianMarkovRandomFields
using LinearAlgebra
using Ferrite

@testset "MaternModel" begin
    @testset "Direct Constructor" begin
        # Create a simple 2D grid for testing
        grid = generate_grid(Triangle, (2, 2))
        interpolation = Lagrange{RefTriangle, 1}()
        quadrature = QuadratureRule{RefTriangle}(2)
        discretization = FEMDiscretization(grid, interpolation, quadrature)

        # Valid construction
        model = MaternModel(discretization; smoothness = 2)
        @test model.discretization === discretization
        @test model.smoothness == 2

        # Test type parameters - now F<:FEMDiscretization, S<:Integer
        @test model isa MaternModel{<:FEMDiscretization, Int}

        # Invalid smoothness
        @test_throws ArgumentError MaternModel(discretization; smoothness = -1)
        @test_throws ArgumentError MaternModel(discretization; smoothness = -2)

        # Zero smoothness should be valid
        model_zero = MaternModel(discretization; smoothness = 0)
        @test model_zero.smoothness == 0
    end

    @testset "Automatic Constructor - 2D" begin
        # Simple 2D triangle of points as N×2 matrix
        points = [0.0 0.0; 1.0 0.0; 0.5 1.0]

        # Valid construction with defaults
        model = MaternModel(points; smoothness = 1)
        @test model isa MaternModel{<:FEMDiscretization, Int}
        @test model.smoothness == 1
        @test length(model) > 0  # Should have created some DOFs

        # Valid construction with custom parameters
        model2 = MaternModel(points; smoothness = 2, element_order = 1)
        @test model2.smoothness == 2

        # Invalid inputs
        @test_throws ArgumentError MaternModel([0.0 0.0; 1.0 0.0]; smoothness = 1)  # Too few points
        @test_throws ArgumentError MaternModel(points; smoothness = -1)  # Negative smoothness
    end

    @testset "Automatic Constructor - Matrix Format Validation" begin
        # Test wrong matrix dimensions (should be N×2)
        points_wrong = [0.0 0.0 0.0; 1.0 0.0 0.0; 0.5 1.0 0.0]  # N×3
        @test_throws ArgumentError MaternModel(points_wrong; smoothness = 1)

        # Test N×1 matrix (wrong shape)
        points_wrong2 = reshape([0.0, 0.0, 0.5], :, 1)  # 3×1 matrix
        @test_throws ArgumentError MaternModel(points_wrong2; smoothness = 1)

        # Test correct N×2 format should work
        points_correct = [0.0 0.0; 1.0 0.0; 0.5 1.0; 0.2 0.8]
        model = MaternModel(points_correct; smoothness = 1)
        @test model isa MaternModel{<:FEMDiscretization, Int}
    end

    @testset "LatentModel Interface" begin
        # Create simple test model
        grid = generate_grid(Triangle, (1, 1))
        discretization = FEMDiscretization(grid, Lagrange{RefTriangle, 1}(), QuadratureRule{RefTriangle}(2))
        model = MaternModel(discretization; smoothness = 1)

        @testset "Basic Interface" begin
            # length method
            n = length(model)
            @test n isa Int
            @test n > 0

            # hyperparameters method
            params = hyperparameters(model)
            @test params == (range = Real,)

            # model_name method
            @test model_name(model) == :matern
        end

        @testset "Parameter Validation" begin
            # Valid range should work
            @test precision_matrix(model; range = 1.0) isa AbstractMatrix
            @test precision_matrix(model; range = 0.1) isa AbstractMatrix
            @test precision_matrix(model; range = 10.0) isa AbstractMatrix

            # Invalid range should throw
            @test_throws ArgumentError precision_matrix(model; range = 0.0)
            @test_throws ArgumentError precision_matrix(model; range = -1.0)
            @test_throws ArgumentError precision_matrix(model; range = -0.1)
        end

        @testset "Mean Vector" begin
            μ = mean(model)
            @test μ isa AbstractVector
            @test length(μ) == length(model)
            @test all(μ .== 0.0)  # Should be zero mean

            # Parameters shouldn't affect mean
            μ2 = mean(model; range = 2.0)
            @test μ2 == μ
        end

        @testset "Constraints" begin
            # Matérn models should have no constraints
            constraint_info = constraints(model; range = 1.0)
            @test constraint_info === nothing

            # Different parameters shouldn't affect constraints
            constraint_info2 = constraints(model; range = 5.0)
            @test constraint_info2 === nothing
        end

        @testset "Precision Matrix Properties" begin
            range_vals = [0.5, 1.0, 2.0]

            for range in range_vals
                Q = precision_matrix(model; range = range)

                # Should be square matrix
                @test size(Q, 1) == size(Q, 2)
                @test size(Q, 1) == length(model)

                # Should be symmetric (approximately, due to numerical precision)
                Q_mat = Matrix(Q)
                @test Q_mat ≈ Q_mat' atol = 1.0e-12

                # Should be positive definite (all eigenvals > 0)
                eigs = real.(eigvals(Q_mat))
                @test all(eigs .> 0)
            end
        end
    end

    @testset "GMRF Construction" begin
        grid = generate_grid(Triangle, (1, 1))
        discretization = FEMDiscretization(grid, Lagrange{RefTriangle, 1}(), QuadratureRule{RefTriangle}(2))
        model = MaternModel(discretization; smoothness = 1)

        # Construct GMRF
        range = 2.0
        gmrf = model(range = range)

        @test gmrf isa GMRF  # Should return GMRF, not ConstrainedGMRF
        @test length(gmrf) == length(model)
        @test all(mean(gmrf) .== 0.0)

        # Precision matrix should match
        Q_model = precision_matrix(model; range = range)
        Q_gmrf = precision_map(gmrf)
        @test Matrix(Q_model) ≈ Matrix(Q_gmrf)
    end

    @testset "Multiple Construction Modes Consistency" begin
        # Create a model using direct construction
        grid = generate_grid(Triangle, (2, 2))
        discretization = FEMDiscretization(grid, Lagrange{RefTriangle, 1}(), QuadratureRule{RefTriangle}(2))
        model_direct = MaternModel(discretization; smoothness = 1)

        # Create a model using automatic construction with a grid that approximates the same domain
        points = [0.0 0.0; 1.0 0.0; 1.0 1.0; 0.0 1.0]  # Square corners
        model_auto = MaternModel(points; smoothness = 1)

        # Both should work and produce reasonable results
        gmrf_direct = model_direct(range = 1.0)
        gmrf_auto = model_auto(range = 1.0)

        @test gmrf_direct isa GMRF
        @test gmrf_auto isa GMRF
        @test length(gmrf_direct) > 0
        @test length(gmrf_auto) > 0

        # Both should have zero mean
        @test all(mean(gmrf_direct) .== 0.0)
        @test all(mean(gmrf_auto) .== 0.0)
    end

    @testset "Different Smoothness Values" begin
        grid = generate_grid(Triangle, (1, 1))
        discretization = FEMDiscretization(grid, Lagrange{RefTriangle, 1}(), QuadratureRule{RefTriangle}(2))

        smoothness_vals = [0, 1, 2, 3]
        range = 1.0

        for smoothness in smoothness_vals
            model = MaternModel(discretization; smoothness = smoothness)
            @test model.smoothness == smoothness

            # Should be able to construct GMRF
            gmrf = model(range = range)
            @test gmrf isa GMRF
            @test length(gmrf) == length(model)

            # Precision matrix should be positive definite
            Q = precision_map(gmrf)
            eigs = real.(eigvals(Matrix(Q)))
            @test all(eigs .> 0)
        end
    end

    @testset "Range Parameter Effects" begin
        grid = generate_grid(Triangle, (1, 1))
        discretization = FEMDiscretization(grid, Lagrange{RefTriangle, 1}(), QuadratureRule{RefTriangle}(2))
        model = MaternModel(discretization; smoothness = 1)

        ranges = [0.1, 0.5, 1.0, 2.0, 5.0]
        precision_matrices = []

        for range in ranges
            Q = precision_matrix(model; range = range)
            push!(precision_matrices, Q)

            # All should be valid precision matrices
            Q_mat = Matrix(Q)
            eigs = real.(eigvals(Q_mat))
            @test all(eigs .> 0)  # Positive definite
        end

        # Different ranges should give different precision matrices
        for i in 2:length(precision_matrices)
            @test !(Matrix(precision_matrices[i]) ≈ Matrix(precision_matrices[1]))
        end
    end

    @testset "Type Stability" begin
        grid = generate_grid(Triangle, (1, 1))
        discretization = FEMDiscretization(grid, Lagrange{RefTriangle, 1}(), QuadratureRule{RefTriangle}(2))
        model = MaternModel(discretization; smoothness = 1)

        # Test with different numeric types
        Q_float64 = precision_matrix(model; range = 1.0)
        Q_int = precision_matrix(model; range = 1)

        @test eltype(Matrix(Q_float64)) == Float64
        @test eltype(Matrix(Q_int)) == Float64  # Should promote

        # GMRF construction should be type stable
        gmrf = model(range = 1.0)
        @test gmrf isa GMRF{Float64}
    end

    @testset "1D MaternModel" begin
        # Test 1D points - need to reshape as N×1 matrix
        points_1d = reshape(sort!(rand(100) * 10.0), :, 1)  # 100×1 matrix

        # Should throw error for 1D points (current implementation expects 2D)
        @test_throws ArgumentError MaternModel(points_1d; smoothness = 1)

        # Test with 1D grid directly using FEMDiscretization
        grid_1d = generate_grid(Line, (10,))
        interpolation_1d = Lagrange{RefLine, 1}()
        quadrature_1d = QuadratureRule{RefLine}(2)
        discretization_1d = FEMDiscretization(grid_1d, interpolation_1d, quadrature_1d)

        model_1d = MaternModel(discretization_1d; smoothness = 1)
        @test model_1d isa MaternModel{<:FEMDiscretization, Int}
        @test model_1d.smoothness == 1
        @test length(model_1d) > 0

        # Test GMRF construction in 1D
        gmrf_1d = model_1d(range = 2.0)
        @test gmrf_1d isa GMRF
        @test length(gmrf_1d) == length(model_1d)
        @test all(mean(gmrf_1d) .== 0.0)

        # Test precision matrix properties in 1D
        Q_1d = precision_matrix(model_1d; range = 1.0)
        Q_1d_mat = Matrix(Q_1d)
        @test size(Q_1d_mat, 1) == length(model_1d)
        @test Q_1d_mat ≈ Q_1d_mat'  # Should be symmetric
        eigs_1d = real.(eigvals(Q_1d_mat))
        @test all(eigs_1d .> 0)  # Should be positive definite
    end
end
