using GaussianMarkovRandomFields
using LinearAlgebra, SparseArrays, Random
using ReTest

@testset "PermutedMatrix" begin
    @testset "Constructor validation" begin
        A = [1 2 3; 4 5 6; 7 8 9]
        p = [3, 1, 2]
        PM = PermutedMatrix(A, p)
        @test PM isa PermutedMatrix

        # Error cases
        @test_throws ArgumentError PermutedMatrix([1 2; 3 4; 5 6], [1, 2, 3])  # Non-square
        @test_throws ArgumentError PermutedMatrix(A, [1, 2])  # Length mismatch
        @test_throws ArgumentError PermutedMatrix(A, [1, 1, 2])  # Invalid permutation
    end

    @testset "Indexing correctness" begin
        A = [1 2 3; 4 5 6; 7 8 9]
        p = [3, 1, 2]
        PM = PermutedMatrix(A, p)

        # Scalar indexing
        @test PM[1, 1] == A[3, 3] == 9
        @test PM[1, 2] == A[3, 1] == 7

        # Vector indexing
        @test PM[1:2, 1:2] == A[p[1:2], p[1:2]]
        @test PM[[1, 3], [2, 3]] == A[p[[1, 3]], p[[2, 3]]]

        # Colon indexing
        @test PM[:, 1] == A[p, p[1]]
        @test PM[1, :] == A[p[1], p]
        @test PM[:, :] == A[p, p]

        # Bounds checking
        @test_throws BoundsError PM[0, 1]
        @test_throws BoundsError PM[1, 4]
    end

    @testset "Edge cases" begin
        # 1x1 matrix
        PM1 = PermutedMatrix(fill(42.0, 1, 1), [1])
        @test PM1[1, 1] == 42.0

        # Identity permutation
        A = rand(3, 3)
        PM_id = PermutedMatrix(A, [1, 2, 3])
        @test PM_id[:, :] == A

        # Sparse matrix
        S = sparse([1 2; 3 4])
        PS = PermutedMatrix(S, [2, 1])
        @test PS[:, :] == S[[2, 1], [2, 1]]
    end

    @testset "to_matrix" begin
        A = [1 2 3; 4 5 6; 7 8 9]
        p = [3, 1, 2]
        PM = PermutedMatrix(A, p)
        @test to_matrix(PM) == A[p, p]
    end
end
