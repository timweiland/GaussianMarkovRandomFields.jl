using GaussianMarkovRandomFields, LinearAlgebra, SparseArrays, Random, Distributions

@testset "ChordalGMRF" begin
    n = 6
    Q = spdiagm(-1 => fill(-1.0, n - 1), 0 => fill(2.5, n), 1 => fill(-1.0, n - 1))
    μ = collect(range(-1.0, 1.0; length = n))
    d = ChordalGMRF(μ, Q)

    @testset "Construction from an existing factorization" begin
        F = cholesky!(GaussianMarkovRandomFields.ChordalCholesky(Hermitian(Q, :L)))
        d_F = ChordalGMRF(μ, Q, F)
        @test mean(d_F) == μ
        @test logdetcov(d_F) ≈ -logdet(Matrix(Q))
    end

    @testset "Precision and gradient" begin
        @test Matrix(precision_map(d)) ≈ Matrix(Q)
        x = randn(MersenneTwister(3), n)
        @test gradlogpdf(d, x) ≈ Q * (μ - x)
    end

    @testset "Sampling" begin
        # A sample is μ + L⁻ᵀz for the Cholesky factor L of Q, so its Mahalanobis
        # distance is exactly ‖z‖² for the same standard normal draws z.
        z = randn(MersenneTwister(1), n)
        @test sqmahal(d, rand(MersenneTwister(1), d)) ≈ dot(z, z)

        x = zeros(n)
        rand!(MersenneTwister(1), d, x)
        @test sqmahal(d, x) ≈ dot(z, z)

        # Several samples at once fill the columns of a preallocated matrix
        X = rand(MersenneTwister(1), d, 3)
        @test sqmahal(d, X[:, 1]) ≈ dot(z, z)
        @test all(sqmahal(d, X[:, j]) > 0 for j in 1:3)
        @test X[:, 2] != X[:, 3]

        rng = MersenneTwister(2611)
        samples = reduce(hcat, [rand(rng, d) for _ in 1:20_000])
        @test vec(mean(samples; dims = 2)) ≈ μ atol = 0.05
        @test cov(samples; dims = 2) ≈ inv(Matrix(Q)) atol = 0.05
    end

    @testset "show" begin
        @test sprint(show, d) == "ChordalGMRF{Float64}(n=6)"
        s = repr(MIME"text/plain"(), d)
        @test startswith(s, "ChordalGMRF{Float64} with 6 variables")
        @test occursin("Mean: [", s)
        d_long = ChordalGMRF(collect(1.0:10.0), sparse(2.0I, 10, 10))
        @test occursin("...", repr(MIME"text/plain"(), d_long))
    end
end
