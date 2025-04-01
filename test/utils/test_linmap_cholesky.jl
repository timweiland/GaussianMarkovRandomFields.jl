using GaussianMarkovRandomFields
using Random
using SparseArrays
using LinearAlgebra
using LinearMaps

@testset "Linmap Cholesky" begin
    rng = MersenneTwister(6398324)
    N = 100
    A = sprand(rng, N, N, 0.3)
    Q = A * A' + 0.1 * I
    Q_map = LinearMap(Q)
    Q_map_cho = linmap_cholesky(Q_map)
    Q_mat_cho = cholesky(Q)
    @test Q_map_cho isa SparseArrays.CHOLMOD.Factor
    for k in 1:3
        x = rand(N)
        @test Q_map_cho \ x ≈ Q_mat_cho \ x
    end

    p = Q_mat_cho.p
    Q_map_cho_same_p = linmap_cholesky(Q_map; perm=p)
    @test Q_map_cho_same_p.p == Q_mat_cho.p
    @test sparse(Q_map_cho_same_p.L) ≈ sparse(Q_mat_cho.L)
end
