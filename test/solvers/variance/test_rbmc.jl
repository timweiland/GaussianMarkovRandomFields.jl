using Distributions, Ferrite, Random

function _generate_gmrf()
    grid = generate_grid(Triangle, (20, 20))
    ip = Lagrange{RefTriangle,1}()
    qr = QuadratureRule{RefTriangle}(2)
    disc = FEMDiscretization(grid, ip, qr)
    spde = MaternSPDE{2}(range = 0.2, smoothness = 1)
    return discretize(spde, disc)
end

@testset "RBMC Variance computations" begin
    rng = MersenneTwister(854289)

    x = _generate_gmrf()
    x_taka = GMRF(
        mean(x),
        precision_map(x),
        CholeskySolverBlueprint(var_strategy = TakahashiStrategy()),
    )
    var_true = var(x_taka)

    x_rbmc = GMRF(
        mean(x),
        precision_map(x),
        CholeskySolverBlueprint(var_strategy = RBMCStrategy(500, rng = rng)),
    )
    var_rbmc = var(x_rbmc)

    # Less than 5% relative error can be expected for this amount of samples
    @test norm(var_rbmc - var_true) / norm(var_true) < 0.05

    block_bp = CholeskySolverBlueprint(
        var_strategy = BlockRBMCStrategy(500, rng = rng, enclosure_size = 2),
    )
    x_block_rbmc = GMRF(mean(x), precision_map(x), block_bp)
    var_block_rbmc = var(x_block_rbmc)

    # Less than 1% relative error can be expected for this BlockRBMC setup
    @test norm(var_block_rbmc - var_true) / norm(var_true) < 0.01
end
