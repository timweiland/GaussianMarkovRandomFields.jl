using GaussianMarkovRandomFields, Ferrite

@testset "Mesh Utils" begin
    N_xy = 20

    @testset "Inflated rectangle, order $d" for d ∈ [1, 2]
        dx = dy = 2.0
        dboundary = 1
        Δ_int = 0.1
        Δ_ext = 0.5
        grid = create_inflated_rectangle(
            0,
            0,
            dx,
            dy,
            dboundary,
            Δ_int,
            Δ_ext;
            element_order = d,
        )

        @test all(
            n.x[1] >= -dboundary &&
            n.x[1] <= (dx + dboundary) &&
            n.x[2] >= -dboundary &&
            n.x[2] <= (dy + dboundary) for n in grid.nodes
        )

        is_inside = (x, y) -> (x >= 0 && x <= dx && y >= 0 && y <= dy)
        N_inside = count(n -> is_inside(n.x[1], n.x[2]), grid.nodes)
        N_outside = length(grid.nodes) - N_inside
        @test N_inside > N_outside
    end
end
