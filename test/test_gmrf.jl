using GMRFs
using Distributions
using LinearAlgebra
using SparseArrays

@testset "GMRF" begin
    μ_standard = zeros(3)
    Q_standard = spdiagm(0 => [1.0, 1.0, 1.0])
    μ_diag = [5.0, 6.0, 7.0, 8.0]
    Q_diag = spdiagm(0 => [1.0, 2.0, 3.0, 4.0])
    d_standard = GMRF(μ_standard, Q_standard)
    d_diag = GMRF(μ_diag, Q_diag)
    d_diag_low_noise = GMRF(μ_diag, 1e10 * Q_diag)

    @testset "Basic quantities" begin
        @test length(d_standard) == 3
        @test length(d_diag) == 4

        @test mean(d_standard) == μ_standard
        @test mean(d_diag) == μ_diag
        @test precision_mat(d_standard) == Q_standard
        @test precision_mat(d_diag) == Q_diag

        @test invcov(d_standard) == Q_standard
        @test invcov(d_diag) == Q_diag
    end

    @test cov(d_standard) ≈ inv(Array(Q_standard))
    @test cov(d_diag) ≈ inv(Array(Q_diag))

    @test logdetcov(d_standard) ≈ -logdet(Q_standard)
    @test logdetcov(d_diag) ≈ -logdet(Q_diag)

    @testset "Sampling" begin
        low_noise_samples = rand(d_diag_low_noise, (10,))
        for sample in low_noise_samples
            @test all(abs.(sample .- μ_diag) .< 1e-4)
            @test sample != μ_diag
        end
    end

    @testset "Squared Mahalanobis distance" begin
        x = rand(3)
        @test sqmahal(d_standard, x) ≈ dot(x, x)
        x = rand(4)
        for d in [d_diag, d_diag_low_noise]
            @test sqmahal(d, x) ≈ dot(x - mean(d), precision_mat(d) * (x - mean(d)))
            @test sqmahal(d, mean(d)) ≈ 0.0
        end
    end

    @testset "Log PDF gradient" begin
        # TODO: Add further tests?
        for d in [d_standard, d_diag, d_diag_low_noise]
            @test gradlogpdf(d, mean(d)) ≈ zeros(length(d))
        end
    end
end