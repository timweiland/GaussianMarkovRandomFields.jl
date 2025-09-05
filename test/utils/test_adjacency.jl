using ReTest
using SparseArrays
using LibGEOS
using GaussianMarkovRandomFields

@testset "Adjacency (queen contiguity)" begin
    # Three unit boxes: [0,1]x[0,1], [1,2]x[0,1] (touch at edge), [3,4]x[0,1] (disjoint)
    g1 = readgeom("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))")
    g2 = readgeom("POLYGON((1 0, 2 0, 2 1, 1 1, 1 0))")
    g3 = readgeom("POLYGON((3 0, 4 0, 4 1, 3 1, 3 0))")

    W = contiguity_adjacency([g1, g2, g3])
    @test size(W) == (3, 3)
    @test W[1, 2] == 1.0 && W[2, 1] == 1.0
    @test W[1, 3] == 0.0 && W[2, 3] == 0.0
    @test all(diag(W) .== 0.0)
end

@testset "Adjacency from GeometryCollection" begin
    # Two touching squares and one distant in a GeometryCollection
    gc = readgeom(
        "GEOMETRYCOLLECTION("
            * "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0)),"
            * "POLYGON((1 0, 2 0, 2 1, 1 1, 1 0)),"
            * "POLYGON((3 0, 4 0, 4 1, 3 1, 3 0))"
            * ")"
    )

    @test gc isa LibGEOS.GeometryCollection
    Wc = contiguity_adjacency(gc)
    @test size(Wc) == (3, 3)
    @test Wc[1, 2] == 1.0 && Wc[2, 1] == 1.0
    @test Wc[1, 3] == 0.0 && Wc[2, 3] == 0.0
end
