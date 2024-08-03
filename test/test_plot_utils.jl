using GMRFs, Ferrite

@testset "Plot Utils" begin
    grid = generate_grid(Triangle, (20, 20)) # Defaults to [-1, 1] x [-1, 1]

    node_idcs = grid.cells[1].nodes
    node_coords = [n.x for n in grid.nodes[[node_idcs...]]]
    @test Set(get_coords(grid, 1)) == Set(node_coords)

    bounds = [[-0.5, 0.5], [-0.5, 0.5]]
    @test in_bounds([0.0, 0.0], bounds) == true
    @test in_bounds([0.6, 0.0], bounds) == false

    x_min = minimum([n[1] for n in node_coords])
    x_max = maximum([n[1] for n in node_coords])
    y_min = minimum([n[2] for n in node_coords])
    y_max = maximum([n[2] for n in node_coords])

    @test in_bounds_cell(grid, [(x_min - 0.1, x_max + 0.1), (y_min - 0.1, y_max + 0.1)], 1)
    @test !in_bounds_cell(grid, [(x_min - 0.1, x_max - 0.1), (y_min - 0.1, y_max - 0.1)], 1)
end
