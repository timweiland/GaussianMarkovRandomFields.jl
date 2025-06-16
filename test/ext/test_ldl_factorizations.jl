using Ferrite
using Random
using LDLFactorizations

function _get_discretization()
    grid = generate_grid(Triangle, (20, 20))
    ip = Lagrange{RefTriangle,1}()
    qr = QuadratureRule{RefTriangle}(2)
    return FEMDiscretization(grid, ip, qr)
end

function _generate_gmrf(disc, method)
    spde = MaternSPDE{2}(range = 0.2, smoothness = 1)
    return discretize(
        spde,
        disc,
        solver_blueprint=CholeskySolverBlueprint{method}()
    )
end

@testset "LDLFactorizations.jl" begin
    construct_rng = () -> MersenneTwister(2349882394)
    #rng = MersenneTwister(seed)

    disc = _get_discretization()
    x_default = _generate_gmrf(disc, :default)
    x_ldl = _generate_gmrf(disc, :autodiffable)

    A = evaluation_matrix(disc, [Tensors.Vec(0.33, 0.44)])
    y = [0.42]
    noise = 1e8

    x_cond_default = condition_on_observations(x_default, A, noise, y)
    x_cond_ldl = condition_on_observations(x_ldl, A, noise, y)

    @testset "Fundamental computations" begin
        @test std(x_ldl) ≈ std(x_default)
        @test logdetcov(x_ldl) ≈ logdetcov(x_default)

        @test x_cond_default.solver isa LinearConditionalCholeskySolver{:default}
        @test x_cond_ldl.solver isa LinearConditionalCholeskySolver{:autodiffable}
        @test mean(x_cond_ldl) ≈ mean(x_cond_default)
        @test std(x_cond_ldl) ≈ std(x_cond_default)
        @test logdetcov(x_cond_ldl) ≈ logdetcov(x_cond_default)
        @test logpdf(x_cond_ldl, mean(x_cond_ldl)) ≈ logpdf(x_cond_default, mean(x_cond_default))
    end

    @testset "Autodiff" begin
        using ForwardDiff

        function form_gmrf(range)
            grid = generate_grid(Triangle, (10, 10))  # Smaller grid for faster tests
            interpolation_fn = Lagrange{RefTriangle, 1}()
            quad_rule = QuadratureRule{RefTriangle}(2)
            disc = FEMDiscretization(grid, interpolation_fn, quad_rule)

            # Define SPDE and discretize to get a GMRF
            spde = MaternSPDE{2}(range = range, smoothness = 2)
            cbp = CholeskySolverBlueprint{:autodiffable}()
            return discretize(spde, disc, solver_blueprint=cbp)
        end

        @testset "Gradient w.r.t. range parameter" begin
            # Test gradient of logdetcov w.r.t range
            logdetcov_fn = range -> logdetcov(form_gmrf(range))
            grad = ForwardDiff.derivative(logdetcov_fn, 0.2)
            @test isa(grad, Real)
            @test isfinite(grad)

            # Test gradient of marginal variance w.r.t range  
            var_fn = range -> var(form_gmrf(range))[1]  # First element
            grad_var = ForwardDiff.derivative(var_fn, 0.2)
            @test isa(grad_var, Real)
            @test isfinite(grad_var)
        end

        @testset "Gradient of logpdf" begin
            x_autodiff = form_gmrf(0.2)
            sample_point = mean(x_autodiff)
            
            # Test gradient of logpdf w.r.t. range parameter
            logpdf_fn = range -> begin
                gmrf = form_gmrf(range)
                logpdf(gmrf, sample_point)
            end
            
            grad = ForwardDiff.derivative(logpdf_fn, 0.2)
            @test isa(grad, Real)
            @test isfinite(grad)
        end

        @testset "Gradient through conditioning" begin
            # Test autodiff through conditional GMRF operations
            function conditional_logpdf(range)
                gmrf = form_gmrf(range)
                
                # Create observation operator
                grid = generate_grid(Triangle, (10, 10))
                ip = Lagrange{RefTriangle,1}()
                qr = QuadratureRule{RefTriangle}(2)
                disc = FEMDiscretization(grid, ip, qr)
                
                A = evaluation_matrix(disc, [Tensors.Vec(0.3, 0.3)])
                y = [0.5]
                noise = 1e-2
                
                x_cond = condition_on_observations(gmrf, A, noise, y)
                return logpdf(x_cond, mean(x_cond))
            end
            
            grad = ForwardDiff.derivative(conditional_logpdf, 0.2)
            @test isa(grad, Real)
            @test isfinite(grad)
        end

        @testset "Gradient of mean computation" begin
            # Test gradient through mean computation in conditional GMRF
            function conditional_mean_sum(range)
                gmrf = form_gmrf(range)
                
                grid = generate_grid(Triangle, (10, 10))
                ip = Lagrange{RefTriangle,1}()
                qr = QuadratureRule{RefTriangle}(2)
                disc = FEMDiscretization(grid, ip, qr)
                
                A = evaluation_matrix(disc, [Tensors.Vec(0.3, 0.3)])
                y = [0.5]
                noise = 1e-2
                
                x_cond = condition_on_observations(gmrf, A, noise, y)
                return sum(mean(x_cond))
            end
            
            grad = ForwardDiff.derivative(conditional_mean_sum, 0.2)
            @test isa(grad, Real)
            @test isfinite(grad)
        end
    end
end
