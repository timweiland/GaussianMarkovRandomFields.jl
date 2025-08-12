using GaussianMarkovRandomFields, LinearAlgebra, LinearMaps, Random, SparseArrays

@testset "CholeskyFactorizedMap" begin
    rng = MersenneTwister(364872)
    N = 25

    A = sprand(rng, N, N, 0.2)
    A = A * A' + I
    A_cho = cholesky(A)

    A_cho_map = CholeskyFactorizedMap(A_cho)

    x = rand(N)

    # Test that we can recover the original factorization
    @test A_cho_map.cho == A_cho
    @test size(A_cho_map) == size(A_cho) == (N, N)
    @test A_cho_map * x ≈ A * x
    @test A_cho_map' == A_cho_map
    @test to_matrix(A_cho_map) ≈ A
    @test issymmetric(A_cho_map)
    @test isposdef(A_cho_map)
end
