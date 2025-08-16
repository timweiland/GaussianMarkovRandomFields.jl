using LinearAlgebra, LinearMaps, Random, SparseArrays

@testset "Symmetric block tridiagonal map" begin
    rng = MersenneTwister(782589312943)

    @testset "Trivial case" begin
        N = 5
        A = rand(rng, N, N)
        A = Symmetric(A * A')
        L = SymmetricBlockTridiagonalMap((LinearMap(A),), ())
        @test size(L) == size(A)
        @test issymmetric(L)
        L_tomat = to_matrix(L)
        L_mat = sparse(A)
        @test parent(L_tomat) isa SparseMatrixCSC
        @test L_tomat ≈ L_mat

        for i in 1:5
            x = rand(rng, N)
            @test L * x ≈ L_mat * x
        end
    end

    @testset "3x3 case" begin
        N₁, N₂, N₃ = 5, 4, 3
        N_total = N₁ + N₂ + N₃
        diags = [rand(rng, N₁, N₁), rand(rng, N₂, N₂), rand(rng, N₃, N₃)]
        diags = [Symmetric(b * b') for b in diags]
        off_diags = [rand(rng, N₂, N₁), rand(rng, N₃, N₂)]
        diag_maps = [LinearMap(b) for b in diags]
        off_diag_maps = [LinearMap(b) for b in off_diags]

        L_mat = sparse(
            [
                diags[1] off_diags[1]' zeros(N₁, N₃)
                off_diags[1] diags[2] off_diags[2]'
                zeros(N₃, N₁) off_diags[2] diags[3]
            ],
        )
        L = SymmetricBlockTridiagonalMap(
            Tuple(LinearMap(b) for b in diags),
            Tuple(LinearMap(b) for b in off_diags),
        )
        @test size(L) == (N_total, N_total)
        @test issymmetric(L)
        L_tomat = to_matrix(L)
        @test parent(L_tomat) isa SparseMatrixCSC
        @test L_tomat ≈ L_mat

        for i in 1:5
            x = rand(rng, N_total)
            @test L * x ≈ L_mat * x
        end
    end
end
