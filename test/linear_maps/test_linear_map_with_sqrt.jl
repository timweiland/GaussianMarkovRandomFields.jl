using GMRFs, LinearMaps, LinearAlgebra, Random

@testset "Linear map with sqrt" begin
    rng = MersenneTwister(823498)
    N, M = 4, 6
    L = rand(rng, N, M)
    A = L * L'
    x = rand(rng, N)
    A_map, L_map = LinearMap(A), LinearMap(L)
    A_with_sqrt = LinearMapWithSqrt(A_map, L_map)

    @test size(A_with_sqrt) == size(A_map) == size(A)
    @test A_with_sqrt * x ≈ A * x
    @test linmap_sqrt(A_with_sqrt) === L_map
    @test to_matrix(A_with_sqrt) ≈ A
    @test issymmetric(A_with_sqrt)
    @test A_with_sqrt' === A_with_sqrt
end
