using LinearAlgebra, LinearMaps, Random, SparseArrays

@testset "SSM Bidiagonal map" begin
    rng = MersenneTwister(7832954811)

    A = sprand(rng, 4, 3, 0.3)
    B = sprand(rng, 3, 3, 0.3)
    C = sprand(rng, 3, 3, 0.3)
    Z_A = spzeros(4, 3)
    Z = spzeros(3, 3)
    N_t = 4

    L_mat = [
        A Z_A Z_A
        B C Z
        Z B C
        Z Z B
    ]
    A_map, B_map, C_map = (LinearMap(X) for X in (A, B, C))
    L = SSMBidiagonalMap(A_map, B_map, C_map, N_t)
    @test size(L) == size(L_mat)
    for i = 1:5
        x = rand(rng, size(L, 2))
        y = rand(rng, size(L, 1))
        @test L * x ≈ L_mat * x
        @test L' * y ≈ L_mat' * y
    end
end
