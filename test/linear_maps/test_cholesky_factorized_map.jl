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

@testset "CholeskyFactorizedMap show" begin
    C = CholeskyFactorizedMap(cholesky(sparse(2.0I, 3, 3)))
    @test startswith(sprint(show, C), "3x3 CholeskyFactorizedMap")
end

@testset "CholeskyFactorizedMap with a dense factor" begin
    A = [4.0 1.0 0.0; 1.0 3.0 1.0; 0.0 1.0 2.0]
    C = CholeskyFactorizedMap(cholesky(A))
    @test to_matrix(C) ≈ A
    @test transpose(C) === C
    @test linmap_sqrt(C) === C.sqrt_map
    @test GaussianMarkovRandomFields.linmap_cholesky(Val(:default), C) === C.cho
    @test_logs (:warn, r"User-specified permutation") GaussianMarkovRandomFields.linmap_cholesky(
        Val(:default), C; perm = [3, 2, 1]
    )

    # Generic constructor, for factorization types without a dedicated method
    C_generic = CholeskyFactorizedMap{Float64}(cholesky(A))
    @test C_generic * ones(3) ≈ A * ones(3)
end
