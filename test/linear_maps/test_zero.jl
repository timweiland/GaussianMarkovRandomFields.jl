using GaussianMarkovRandomFields, SparseArrays

@testset "Zero map" begin
    Ns, Ms = [10, 20], [10, 15]

    for (N, M) in zip(Ns, Ms)
        Z = ZeroMap{Float64}(N, M)
        @test size(Z) == (N, M)
        Zᵀ = transpose(Z)
        @test size(Zᵀ) == (M, N)
        @test Zᵀ isa ZeroMap{Float64}

        @test to_matrix(Z) ≈ spzeros(N, M)
        Z_sqrt = linmap_sqrt(Z)
        @test Z_sqrt isa ZeroMap{Float64}
        @test size(Z_sqrt) == (N, M)
    end
end
