using GaussianMarkovRandomFields
using LinearAlgebra, Random
using NearestNeighbors
using ReTest

@testset "reverse_maximin_ordering" begin
    @testset "Basic properties" begin
        rng = MersenneTwister(42)
        X = rand(rng, 2, 10)  # 2D points

        P, ℓ = reverse_maximin_ordering(X)

        @test length(P) == size(X, 2)
        @test length(ℓ) == size(X, 2)
        @test isperm(P)
        @test all(ℓ .>= 0)
    end

    @testset "Ordering properties" begin
        # Regular 1D grid
        X = reshape(0.0:0.1:1.0, 1, :)
        P, ℓ = reverse_maximin_ordering(X)

        @test isperm(P)
        # First selected point (last in reverse order) should have largest ℓ
        @test ℓ[P[end]] == maximum(ℓ)
    end

    @testset "Different dimensions" begin
        rng = MersenneTwister(123)

        # 1D
        X1 = reshape(rand(rng, 5), 1, :)
        P1, ℓ1 = reverse_maximin_ordering(X1)
        @test isperm(P1) && length(P1) == 5

        # 2D
        X2 = rand(rng, 2, 8)
        P2, ℓ2 = reverse_maximin_ordering(X2)
        @test isperm(P2) && length(P2) == 8

        # 3D
        X3 = rand(rng, 3, 6)
        P3, ℓ3 = reverse_maximin_ordering(X3)
        @test isperm(P3) && length(P3) == 6
    end

    @testset "Edge cases" begin
        # Single point
        X1 = rand(2, 1)
        P1, ℓ1 = reverse_maximin_ordering(X1)
        @test P1 == [1]
        @test ℓ1[1] == Inf || ℓ1[1] >= 1.0e10  # Very large for first point

        # Two points
        X2 = [0.0 1.0; 0.0 0.0]
        P2, ℓ2 = reverse_maximin_ordering(X2)
        @test isperm(P2)
        @test length(P2) == 2
    end

    @testset "Precomputed tree" begin
        rng = MersenneTwister(456)
        X = rand(rng, 2, 10)

        tree = KDTree(X)
        P1, ℓ1 = reverse_maximin_ordering(X; point_tree = tree)
        P2, ℓ2 = reverse_maximin_ordering(X)

        @test P1 == P2
        @test ℓ1 ≈ ℓ2
    end
end
