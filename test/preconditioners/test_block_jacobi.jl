using GaussianMarkovRandomFields, SparseArrays, LinearAlgebra
import Random: MersenneTwister

@testset "Block Jacobi Preconditioner" begin
    rng = MersenneTwister(36486340)
    N₁ = 25
    N₂ = 20

    A₁ = sprand(rng, N₁, N₁, 0.1)
    A₁ = A₁' * A₁ + 0.1 * I
    A₂ = sprand(rng, N₂, N₂, 0.1)
    A₂ = A₂' * A₂ + 0.1 * I

    P₁ = FullCholeskyPreconditioner(A₁)
    P₂ = FullCholeskyPreconditioner(A₂)
    P = BlockJacobiPreconditioner([P₁, P₂])

    @test size(P) == size(P₁) .+ size(P₂)
    @test size(P, 1) == size(P₁, 1) + size(P₂, 1)

    x = rand(rng, N₁ + N₂)
    y = rand(rng, N₁ + N₂)

    direct_res = [P₁ \ x[1:N₁]; P₂ \ x[(N₁ + 1):end]]
    @test ldiv!(y, P, x) ≈ direct_res
    @test y ≈ direct_res
    @test P \ x ≈ direct_res
    @test ldiv!(P, x) ≈ direct_res
    @test x ≈ direct_res

    @testset "Non-Cholesky blocks" begin
        L = sprand(rng, N₂, N₁, 0.1)
        P_gs = TridiagonalBlockGaussSeidelPreconditioner((A₁, A₂), (L,))
        P_mixed = BlockJacobiPreconditioner([P_gs, P₂])

        @test size(P_gs, 1) == N₁ + N₂
        @test size(P_mixed, 1) == N₁ + 2N₂

        x_mixed = rand(rng, N₁ + 2N₂)
        y_mixed = similar(x_mixed)
        direct_mixed = [P_gs \ x_mixed[1:(N₁ + N₂)]; P₂ \ x_mixed[(N₁ + N₂ + 1):end]]
        @test ldiv!(y_mixed, P_mixed, x_mixed) ≈ direct_mixed
        @test y_mixed ≈ direct_mixed
    end
end
