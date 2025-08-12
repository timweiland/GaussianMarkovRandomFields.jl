using Distributions, Ferrite, Random
using LinearSolve

function _generate_gmrf()
    grid = generate_grid(Triangle, (20, 20))
    ip = Lagrange{RefTriangle,1}()
    qr = QuadratureRule{RefTriangle}(2)
    disc = FEMDiscretization(grid, ip, qr)
    spde = MaternSPDE{2}(range = 0.2, smoothness = 1)
    return GaussianMarkovRandomFields.discretize(spde, disc, algorithm = LinearSolve.CHOLMODFactorization())
end

@testset "RBMC Variance computations" begin
    rng = MersenneTwister(854289)

    x = _generate_gmrf()
    
    # Get true variance using selected inversion
    var_true = var(x)

    # Test RBMC with specific strategy
    x_rbmc = GMRF(
        mean(x),
        precision_map(x),
        LinearSolve.CHOLMODFactorization();
        rbmc_strategy = RBMCStrategy(500, rng = rng)
    )
    var_rbmc = var(x_rbmc, RBMCStrategy(500, rng = rng))

    # Less than 5% relative error can be expected for this amount of samples
    @test norm(var_rbmc - var_true) / norm(var_true) < 0.05

    # Test Block RBMC
    var_block_rbmc = var(x, BlockRBMCStrategy(500, rng = rng, enclosure_size = 2))

    # Less than 1% relative error can be expected for this BlockRBMC setup
    @test norm(var_block_rbmc - var_true) / norm(var_true) < 0.01
end
