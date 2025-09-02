using GaussianMarkovRandomFields, Ferrite
using LinearAlgebra
using SparseArrays
using Random

@testset "Scattered mesh 1D" begin
    rng = MersenneTwister(1249806901)

    X_1d = sort!(rand(rng, 100)) * 10.0
    points_1d = [[x] for x in X_1d]

    grid_1d = generate_mesh(points_1d)
    @test grid_1d isa Ferrite.Grid
    @test length(grid_1d.cells) > 0
    @test length(grid_1d.nodes) > length(points_1d)

    grid_1d_order2 = generate_mesh(points_1d; element_order = 2)
    @test grid_1d_order2 isa Ferrite.Grid
    @test length(grid_1d_order2.cells) > 0
end

@testset "Scattered mesh 2D" begin
    rng = MersenneTwister(1249806901)

    n = 1000
    r = 1.0

    r_cut_start = 0.3
    r_cut_stop = 0.5

    X_train = []
    y_train = Float64[]
    X_test = []
    y_test = Float64[]
    while length(X_train) < n
        x = 2 * r * (rand(rng) - 0.5)
        y = 2 * r * (rand(rng) - 0.5)
        l² = x^2 + y^2
        if l² > r^2
            continue
        elseif (l² < r_cut_start^2 || l² > r_cut_stop^2)
            push!(X_train, [x, y])
            push!(y_train, l² + 0.1 * randn(rng))
        else
            push!(X_test, [x, y])
            push!(y_test, l² + 0.1 * randn(rng))
        end
    end

    X_all = X_train ∪ X_test
    grid = generate_mesh(X_all)
    @test grid isa Ferrite.Grid
    @test length(grid.cells) > 0
    @test length(grid.nodes) > length(X_all)

    ip = Lagrange{RefTriangle, 1}()
    qr = QuadratureRule{RefTriangle}(2)
    disc = FEMDiscretization(grid, ip, qr)

    spde = MaternSPDE{2}(range = 0.7, smoothness = 2)
    u_matern = GaussianMarkovRandomFields.discretize(spde, disc)

    Λ_obs = 10.0
    A_train = evaluation_matrix(disc, [Tensors.Vec(x...) for x in X_train])
    A_test = evaluation_matrix(disc, [Tensors.Vec(x...) for x in X_test])
    u_cond = condition_on_observations(u_matern, A_train, Λ_obs, y_train)

    rmse = (a, b) -> sqrt(mean((a .- b) .^ 2))
    @test rmse(A_test * mean(u_cond), y_test) < 0.25
end
