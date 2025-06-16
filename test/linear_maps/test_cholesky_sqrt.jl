using GaussianMarkovRandomFields, LinearAlgebra, LinearMaps, Random, SparseArrays

@testset "Cholesky Sqrt" begin
    rng = MersenneTwister(98234811)
    N = 10
    A = rand(rng, N, N)
    A = A * A' + I
    A_cho = cholesky(A)

    A_sp = sprand(rng, N, N, 0.2)
    A_sp = A_sp * A_sp' + I
    A_sp_cho = cholesky(A_sp)

    x = rand(N)

    C = CholeskySqrt(A_cho)
    C_sp = CholeskySqrt(A_sp_cho)

    @test size(C) == size(C_sp) == (N, N)
    @test C * (C' * x) ≈ A * x
    @test C_sp * (C_sp' * x) ≈ A_sp * x
end
