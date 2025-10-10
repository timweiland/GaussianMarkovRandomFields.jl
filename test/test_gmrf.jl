using GaussianMarkovRandomFields
using Distributions
using LinearAlgebra
using SparseArrays
using Random

@testset "GMRF" begin
    μ_standard = zeros(3)
    Q_standard = spdiagm(0 => [1.0, 1.0, 1.0])
    μ_diag = [5.0, 6.0, 7.0, 8.0]
    Q_diag = spdiagm(0 => [1.0, 2.0, 3.0, 4.0])
    d_standard = GMRF(μ_standard, Q_standard)
    d_diag = GMRF(μ_diag, Q_diag)
    d_diag_low_noise = GMRF(μ_diag, 1.0e10 * Q_diag)

    rng = Random.MersenneTwister(2359025)

    @testset "Basic quantities" begin
        @test length(d_standard) == 3
        @test length(d_diag) == 4

        @test mean(d_standard) == μ_standard
        @test mean(d_diag) == μ_diag
        @test to_matrix(precision_map(d_standard)) == Q_standard
        @test to_matrix(precision_map(d_diag)) == Q_diag
        @test precision_matrix(d_standard) == Q_standard
        @test precision_matrix(d_diag) == Q_diag

        @test invcov(d_standard) == Q_standard
        @test invcov(d_diag) == Q_diag
    end

    @test_throws ErrorException cov(d_standard)
    @test_throws ErrorException cov(d_diag)

    @test logdetcov(d_standard) ≈ -logdet(Q_standard)
    @test logdetcov(d_diag) ≈ -logdet(Q_diag)

    @testset "Sampling" begin
        low_noise_samples = rand(d_diag_low_noise, (10,))
        for sample in low_noise_samples
            @test all(abs.(sample .- μ_diag) .< 1.0e-4)
            @test sample != μ_diag
        end
    end

    @testset "Squared Mahalanobis distance" begin
        x = rand(3)
        @test sqmahal(d_standard, x) ≈ dot(x, x)
        x = rand(4)
        for d in [d_diag, d_diag_low_noise]
            @test sqmahal(d, x) ≈ dot(x - mean(d), precision_map(d) * (x - mean(d)))
            @test sqmahal(d, mean(d)) ≈ 0.0
        end
    end

    @testset "Log PDF gradient" begin
        # TODO: Add further tests?
        for d in [d_standard, d_diag, d_diag_low_noise]
            @test gradlogpdf(d, mean(d)) ≈ zeros(length(d))
        end
    end

    @testset "Variance and standard deviation" begin
        N = 100
        for i in 1:5
            Q = sprand(N, N, 0.2)
            Q = (Q + Q') / 2 + N * I
            Q⁻¹ = inv(Array(Q))
            Q⁻¹ = (Q⁻¹ + Q⁻¹') / 2
            d = GMRF(zeros(N), Q)
            d2 = MvNormal(Q⁻¹)
            @test var(d) ≈ var(d2)
            @test std(d) ≈ sqrt.(var(d))
        end
    end

    @testset "Type conversions" begin
        # Test type promotion when mean and precision have different types
        μ_f32 = Float32[1.0, 2.0, 3.0]
        Q_f64 = spdiagm(0 => [1.0, 2.0, 3.0])  # Float64

        d = GMRF(μ_f32, Q_f64)
        @test eltype(mean(d)) == Float64
        @test eltype(precision_matrix(d)) == Float64

        # Test with LinearMap
        using LinearMaps
        Q_map = LinearMap(Q_f64)
        d_map = GMRF(μ_f32, Q_map)
        @test eltype(mean(d_map)) == Float64
    end

    @testset "InformationVector constructor" begin
        # Test GMRF construction from information vector
        n = 4
        Q = spdiagm(0 => [2.0, 2.0, 2.0, 2.0])
        μ_expected = [1.0, 2.0, -1.0, 0.5]

        # Create information vector h = Q * μ
        h = Q * μ_expected
        iv = InformationVector(h)

        # Construct GMRF from information vector
        d = GMRF(iv, Q)

        @test mean(d) ≈ μ_expected
        @test information_vector(d) ≈ h

        # Test information_vector accessor
        d_from_mean = GMRF(μ_expected, Q)
        @test information_vector(d_from_mean) ≈ h
    end

    @testset "Linsolve cache reuse" begin
        # Test providing existing linsolve_cache to constructor
        using LinearSolve

        n = 5
        Q = spdiagm(0 => ones(n))
        μ = zeros(n)

        # Create initial GMRF to get a cache
        d1 = GMRF(μ, Q, LinearSolve.CHOLMODFactorization())
        cache1 = d1.linsolve_cache

        # Create new GMRF reusing the cache
        μ2 = ones(n)
        d2 = GMRF(μ2, Q; linsolve_cache = cache1)

        @test mean(d2) == μ2
        @test d2.linsolve_cache === cache1
    end

    @testset "Q_sqrt fallback and errors" begin
        using LinearSolve

        # Test sampling error when Q_sqrt is nothing and algorithm doesn't support backward solve
        # Use KrylovJL_GMRES which doesn't support backward solve
        n = 4
        Q = spdiagm(0 => ones(n))
        μ = zeros(n)

        # This algorithm doesn't support backward solve or selected inversion
        d_no_qsqrt = GMRF(μ, Q, LinearSolve.KrylovJL_GMRES())

        # Should error when trying to sample without Q_sqrt
        @test_throws ErrorException rand(rng, d_no_qsqrt)

        # Should also error when trying to compute variance
        @test_throws ErrorException var(d_no_qsqrt)
    end

    @testset "LDLtFactorization cache update" begin
        # Test that LDLtFactorization properly copies the matrix
        using LinearSolve

        n = 4
        Q = SymTridiagonal(ones(n), fill(-0.5, n - 1))
        μ = zeros(n)

        # Create GMRF with LDLtFactorization
        d = GMRF(μ, Q, LinearSolve.LDLtFactorization())

        # Verify it works correctly
        @test mean(d) == μ
        @test precision_matrix(d) == Q

        # Sample to ensure factorization worked
        samples = rand(rng, d, 5)
        @test size(samples) == (n, 5)
    end
end
