using LinearAlgebra, LinearMaps, Random, SparseArrays

@testset "Tridiagonal block Gauss-Seidel preconditioners" begin
    rng = MersenneTwister(281329)

    D1, D2, D3 = sprand(3, 3, 0.4), sprand(4, 4, 0.4), sprand(3, 3, 0.5)
    D1, D2, D3 = (Symmetric(D * D' + I) for D in (D1, D2, D3))
    D1_p, D2_p, D3_p = FullCholeskyPreconditioner.((D1, D2, D3))
    L1 = sprand(4, 3, 0.6)
    L2 = sprand(3, 4, 0.75)
    P_vanilla_mat = [
        D1 spzeros(3, 4) spzeros(3, 3)
        L1 D2 spzeros(4, 3)
        spzeros(3, 3) L2 D3
    ]
    D_mat = [
        D1 spzeros(3, 4) spzeros(3, 3)
        spzeros(4, 3) D2 spzeros(4, 3)
        spzeros(3, 3) spzeros(3, 4) D3
    ]

    @testset "Vanilla" begin
        P_from_mat = TridiagonalBlockGaussSeidelPreconditioner((D1, D2, D3), (L1, L2))
        P_from_precs = TridiagonalBlockGaussSeidelPreconditioner(
            FullCholeskyPreconditioner.((D1, D2, D3)),
            (L1, L2),
        )
        @test size(P_from_mat) == size(P_vanilla_mat)
        @test size(P_from_precs) == size(P_vanilla_mat)

        P_mat_inv = inv(Array(P_vanilla_mat))
        for i = 1:5
            x = rand(rng, size(P_vanilla_mat, 2))
            gt = P_mat_inv * x
            @test P_from_mat \ x ≈ gt
            @test P_from_precs \ x ≈ gt
            y1 = copy(x)
            ldiv!(P_from_mat, y1)
            @test y1 ≈ gt
            y2 = copy(x)
            ldiv!(P_from_precs, y2)
            @test y2 ≈ gt
        end
    end

    @testset "Symmetric" begin
        P_mat = P_vanilla_mat * inv(Array(D_mat)) * P_vanilla_mat'

        P_from_mat = TridiagSymmetricBlockGaussSeidelPreconditioner((D1, D2, D3), (L1, L2))
        P_from_precs = TridiagSymmetricBlockGaussSeidelPreconditioner(
            FullCholeskyPreconditioner.((D1, D2, D3)),
            (L1, L2),
        )
        @test size(P_from_mat) == size(P_mat)
        @test size(P_from_precs) == size(P_mat)

        P_mat_inv = inv(Array(P_mat))
        for i = 1:5
            x = rand(rng, size(P_mat, 2))
            gt = P_mat_inv * x
            @test P_from_mat \ x ≈ gt
            @test P_from_precs \ x ≈ gt
            y1 = copy(x)
            ldiv!(P_from_mat, y1)
            @test y1 ≈ gt
            y2 = copy(x)
            ldiv!(P_from_precs, y2)
            @test y2 ≈ gt
        end
    end
end
