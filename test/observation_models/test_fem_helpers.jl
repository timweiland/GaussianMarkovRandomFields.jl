using GaussianMarkovRandomFields, Ferrite, Tensors, Distributions, SparseArrays

@testset "FEM Observation Model Helpers" begin
    # Setup common FEM discretization
    N_xy = 10
    grid = generate_grid(Triangle, (N_xy, N_xy))
    ip = Lagrange{RefTriangle, 1}()
    qr = QuadratureRule{RefTriangle}(2)
    fem = FEMDiscretization(grid, ip, qr)

    # Test points
    points = [0.1 0.2; 0.5 0.7; 0.9 0.3]

    @testset "PointEvaluationObsModel" begin
        @testset "Poisson observations" begin
            obs_model = PointEvaluationObsModel(fem, points, Poisson)

            @test obs_model isa LinearlyTransformedObservationModel
            @test size(obs_model.design_matrix) == (3, ndofs(fem))
            @test hyperparameters(obs_model) == ()

            # Test materialization and conditional distribution
            y = [2, 5, 1]
            obs_lik = obs_model(y)
            @test obs_lik isa LinearlyTransformedLikelihood

            x = randn(ndofs(fem))
            dist = conditional_distribution(obs_model, x)
            @test dist isa Distribution
            @test length(dist) == 3
        end

        @testset "Normal observations" begin
            obs_model = PointEvaluationObsModel(fem, points, Normal)

            @test hyperparameters(obs_model) == (:σ,)

            y = [0.1, -0.5, 0.8]
            obs_lik = obs_model(y; σ = 0.2)

            x = randn(ndofs(fem))
            dist = conditional_distribution(obs_model, x; σ = 0.2)
            @test dist isa Distribution
        end
    end

    @testset "PointDerivativeObsModel" begin
        @testset "Single derivative direction" begin
            obs_model = PointDerivativeObsModel(fem, points, Normal; derivative_idcs = [1])

            @test obs_model isa LinearlyTransformedObservationModel
            @test size(obs_model.design_matrix) == (3, ndofs(fem))

            y = [0.1, 0.3, -0.2]
            obs_lik = obs_model(y; σ = 0.1)

            x = randn(ndofs(fem))
            dist = conditional_distribution(obs_model, x; σ = 0.1)
            @test length(dist) == 3
        end

        @testset "Multiple derivative directions" begin
            obs_model = PointDerivativeObsModel(fem, points, Normal; derivative_idcs = [1, 2])

            @test size(obs_model.design_matrix) == (6, ndofs(fem))  # 3 points × 2 derivatives

            y = randn(6)  # [∂u/∂x₁, ∂u/∂x₂, ∂u/∂x₃, ∂u/∂y₁, ∂u/∂y₂, ∂u/∂y₃]
            obs_lik = obs_model(y; σ = 0.1)

            x = randn(ndofs(fem))
            dist = conditional_distribution(obs_model, x; σ = 0.1)
            @test length(dist) == 6
        end

        @testset "Default derivative indices" begin
            obs_model = PointDerivativeObsModel(fem, points, Normal)

            # Should default to all spatial dimensions [1, 2] for 2D
            @test size(obs_model.design_matrix) == (6, ndofs(fem))
        end
    end

    @testset "PointSecondDerivativeObsModel" begin
        # Skip for linear elements since second derivatives are zero
        quad_grid = generate_grid(QuadraticTriangle, (5, 5))
        quad_ip = Lagrange{RefTriangle, 2}()
        quad_fem = FEMDiscretization(quad_grid, quad_ip, qr)

        @testset "Diagonal second derivatives" begin
            obs_model = PointSecondDerivativeObsModel(quad_fem, points, Normal; derivative_idcs = [(1, 1), (2, 2)])

            @test obs_model isa LinearlyTransformedObservationModel
            @test size(obs_model.design_matrix) == (6, ndofs(quad_fem))  # 3 points × 2 derivatives

            y = randn(6)
            obs_lik = obs_model(y; σ = 0.01)

            x = randn(ndofs(quad_fem))
            dist = conditional_distribution(obs_model, x; σ = 0.01)
            @test length(dist) == 6
        end

        @testset "Default second derivative indices" begin
            obs_model = PointSecondDerivativeObsModel(quad_fem, points, Normal)

            # Should default to diagonal terms [(1,1), (2,2)] for 2D
            @test size(obs_model.design_matrix) == (6, ndofs(quad_fem))
        end
    end
end
