using GaussianMarkovRandomFields, SparseArrays, LinearAlgebra
import Random: MersenneTwister

@testset "Full Cholesky Preconditioner" begin
    rng = MersenneTwister(6934345234)
    N = 25
    A = sprand(rng, N, N, 0.1)
    A = A' * A + 0.1 * I

    cho = cholesky(Symmetric(A))
    P = FullCholeskyPreconditioner(cho)
    @test size(P) == size(cho)
    @test size(P, 1) == size(cho, 1)

    x = rand(rng, N)
    y = rand(rng, N)

    cho_res = cho \ x
    @test ldiv!(y, P, x) ≈ cho_res
    @test y ≈ cho_res
    @test P \ x ≈ cho_res
    @test ldiv!(P, x) ≈ cho_res
    @test x ≈ cho_res
end

struct InterfaceOnlyPreconditioner <: GaussianMarkovRandomFields.AbstractPreconditioner{Float64} end

@testset "AbstractPreconditioner interface without implementations" begin
    P = InterfaceOnlyPreconditioner()
    @test_throws MethodError ldiv!(zeros(2), P, ones(2))
    @test_throws MethodError ldiv!(P, ones(2))
    @test_throws MethodError P \ ones(2)
    @test_throws MethodError size(P)
end
