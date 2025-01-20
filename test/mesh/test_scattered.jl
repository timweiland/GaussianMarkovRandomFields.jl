using GMRFs, Ferrite
using LinearAlgebra
using SparseArrays

@testset "Scattered mesh" begin
    # Sample 1000 points within a circle
    n = 1000
    r = 1.0

    r_cut_start = 0.3
    r_cut_stop = 0.5

    X_train = []
    y_train = Float64[]
    X_test = []
    y_test = Float64[]
    while length(X_train) < n
        x = 2 * r * (rand() - 0.5)
        y = 2 * r * (rand() - 0.5)
        l² = x^2 + y^2
        if l² > r^2
            continue
        elseif (l² < r_cut_start^2 || l² > r_cut_stop^2)
            push!(X_train, [x, y])
            push!(y_train, l² + 0.1 * randn())
        else
            push!(X_test, [x, y])
            push!(y_test, l² + 0.1 * randn())
        end
    end

    X_all = X_train ∪ X_test
    grid = generate_mesh(X_all, 0.3, 0.05)
    grid_wider_buffer = generate_mesh(X_all, 0.4, 0.05)
    @test length(grid_wider_buffer.cells) > length(grid.cells)
    grid_smaller_cells = generate_mesh(X_all, 0.3, 0.02)
    @test length(grid_smaller_cells.cells) > length(grid.cells)

    ip = Lagrange{RefTriangle, 1}()
    qr = QuadratureRule{RefTriangle}(2)
    disc = FEMDiscretization(grid, ip, qr)

    spde = MaternSPDE{2}(range=0.9, smoothness=2)
    u_matern = discretize(spde, disc)

    Λ_obs = 10.
    A_train = evaluation_matrix(disc, [Tensors.Vec(x...) for x in X_train])
    A_test = evaluation_matrix(disc, [Tensors.Vec(x...) for x in X_test])
    u_cond = condition_on_observations(u_matern, A_train, Λ_obs, y_train)

    rmse = (a, b) -> sqrt(mean((a .- b).^2))
    @test rmse(A_test * mean(u_cond), y_test) < 0.25
end
