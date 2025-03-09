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
    @test C * x ≈ A_cho.L * x
    @test C' * x ≈ A_cho.U * x
    @test to_matrix(C) ≈ A_cho.L

    p = A_sp_cho.p
    p⁻¹ = invperm(p)
    A_sp_cho_L = sparse(A_sp_cho.L)
    A_sp_cho_L_unpermuted = A_sp_cho_L[p⁻¹, :]

    @test to_matrix(C_sp) ≈ A_sp_cho_L_unpermuted
    @test C_sp * x ≈ A_sp_cho_L_unpermuted * x
    @test C_sp' * x ≈ A_sp_cho_L_unpermuted' * x
end
