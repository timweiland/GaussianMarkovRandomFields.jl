using LinearAlgebra, LinearMaps, Random, SparseArrays

@testset "Test spatiotemporal preconditioner" begin
    rng = MersenneTwister(5820134)

    N_spatial = 3
    D1, D2, D3 = sprand(N_spatial, N_spatial, 0.4),
        sprand(N_spatial, N_spatial, 0.4),
        sprand(N_spatial, N_spatial, 0.5)
    D1, D2, D3 = (Symmetric(D * D' + I) for D in (D1, D2, D3))
    D1_p, D2_p, D3_p = FullCholeskyPreconditioner.((D1, D2, D3))
    L1 = sprand(N_spatial, N_spatial, 0.6)
    L2 = sprand(N_spatial, N_spatial, 0.75)
    A = [
        D1 L1' spzeros(N_spatial, N_spatial)
        L1 D2 L2'
        spzeros(N_spatial, N_spatial) L2 D3
    ]
    P = temporal_block_gauss_seidel(A, N_spatial)
    @test size(P) == size(A)
    @test length(P.D⁻¹_blocks) == 3
    @test length(P.L_blocks) == 2
    @test P.L_blocks[1] ≈ L1
    @test P.L_blocks[2] ≈ L2

    for i in 1:5
        x = rand(rng, N_spatial)
        @test D1 \ x ≈ P.D⁻¹_blocks[1] \ x
        @test D2 \ x ≈ P.D⁻¹_blocks[2] \ x
        @test D3 \ x ≈ P.D⁻¹_blocks[3] \ x
    end
end
