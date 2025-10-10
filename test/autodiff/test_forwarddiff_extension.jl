using GaussianMarkovRandomFields
using ForwardDiff
using Distributions
using Distributions: logdetcov, logpdf
using SparseArrays
using LinearAlgebra
using LinearMaps
using LinearSolve
using Test

@testset "ForwardDiff Extension" begin
    @testset "GMRF construction with Dual numbers - mean only" begin
        # Test GMRF(mean::Vector{Dual}, precision::AbstractMatrix)
        n = 4
        Q = spdiagm(0 => ones(n))

        # Create dual number mean
        θ = [1.0, 2.0]
        function test_mean_dual(θ)
            μ = vcat(θ, zeros(n - length(θ)))
            gmrf = GMRF(μ, Q, LinearSolve.CHOLMODFactorization())
            return sum(mean(gmrf))
        end

        grad = ForwardDiff.gradient(test_mean_dual, θ)
        @test grad ≈ [1.0, 1.0]
    end

    @testset "GMRF construction with Dual numbers - precision only" begin
        # Test GMRF(mean::Vector, precision::AbstractMatrix{Dual})
        n = 3
        μ = zeros(n)

        function test_precision_dual(θ)
            Q = spdiagm(0 => θ)
            gmrf = GMRF(μ, Q, LinearSolve.CHOLMODFactorization())
            return logdetcov(gmrf)
        end

        θ = ones(n)
        grad = ForwardDiff.gradient(test_precision_dual, θ)
        @test all(isfinite.(grad))
    end

    @testset "GMRF construction with Dual numbers - both" begin
        # Test GMRF(mean::Vector{Dual}, precision::AbstractMatrix{Dual})
        n = 3

        function test_both_dual(θ)
            μ = θ[1:n]
            Q = spdiagm(0 => θ[(n + 1):(2 * n)])
            gmrf = GMRF(μ, Q, LinearSolve.CHOLMODFactorization())
            return sum(mean(gmrf)) + logdetcov(gmrf)
        end

        θ = ones(2 * n)
        grad = ForwardDiff.gradient(test_both_dual, θ)
        @test length(grad) == 2 * n
        @test all(isfinite.(grad))
    end

    # Note: LinearMap tests are skipped because ForwardDiff + LinearMaps
    # require special handling that is beyond scope of this test

    @testset "logdetcov with Dual numbers" begin
        # Test specialized logdetcov for GMRF{<:ForwardDiff.Dual}
        n = 4

        function test_logdetcov(θ)
            μ = zeros(n)
            Q = spdiagm(0 => θ)
            gmrf = GMRF(μ, Q, LinearSolve.CHOLMODFactorization())
            return logdetcov(gmrf)
        end

        θ = ones(n) .* 2.0
        grad = ForwardDiff.gradient(test_logdetcov, θ)

        # For diagonal Q with entries q_i, logdetcov = -sum(log(q_i))
        # d/dq_i logdetcov = -1/q_i
        expected = -1.0 ./ θ
        @test grad ≈ expected rtol = 1.0e-6
    end

    @testset "Type conversion helpers (_primal_* functions)" begin
        # Test that _primal_* functions correctly extract primal values from Dual numbers
        # This is tested indirectly: the cache must be built with primal values
        # because LinearSolve caches can't handle Dual numbers
        n = 3

        # Create GMRF with Dual mean - internally calls _primal_mean
        function test_primal_mean(θ)
            μ = θ  # Dual numbers
            Q = spdiagm(0 => ones(n))  # Regular matrix
            gmrf = GMRF(μ, Q, LinearSolve.CHOLMODFactorization())
            # If _primal_mean didn't work, cache creation would fail
            return sum(mean(gmrf))
        end

        θ = ones(n)
        grad1 = ForwardDiff.gradient(test_primal_mean, θ)
        @test grad1 ≈ ones(n)

        # Create GMRF with Dual precision - internally calls _primal_precision variants
        function test_primal_precision_sparse(θ)
            μ = zeros(n)
            Q = spdiagm(0 => θ)  # Dual numbers in sparse matrix
            gmrf = GMRF(μ, Q, LinearSolve.CHOLMODFactorization())
            return logdetcov(gmrf)
        end

        θ2 = ones(n) .* 2.0
        grad2 = ForwardDiff.gradient(test_primal_precision_sparse, θ2)
        @test all(isfinite.(grad2))

        # Test _primal_precision with SymTridiagonal
        function test_primal_precision_symtri(θ)
            μ = zeros(n)
            # Both diagonal and off-diagonal need same element type
            Q = SymTridiagonal(θ, θ[1:(end - 1)] .* 0.0 .+ 0.5)  # Make off-diag also Dual
            gmrf = GMRF(μ, Q, LinearSolve.LDLtFactorization())
            return logdetcov(gmrf)
        end

        grad3 = ForwardDiff.gradient(test_primal_precision_symtri, θ2)
        @test all(isfinite.(grad3))
    end

    @testset "SymTridiagonal with Dual" begin
        # Test with SymTridiagonal precision matrix
        n = 4

        function test_symtri(θ)
            μ = zeros(n)
            Q = SymTridiagonal(θ[1:n], θ[(n + 1):(2 * n - 1)])
            gmrf = GMRF(μ, Q, LinearSolve.LDLtFactorization())
            return logdetcov(gmrf)
        end

        θ = vcat(ones(n) .* 2.0, ones(n - 1) .* 0.5)
        grad = ForwardDiff.gradient(test_symtri, θ)
        @test all(isfinite.(grad))
        @test length(grad) == 2 * n - 1
    end

    @testset "Diagonal with Dual" begin
        # Test with Diagonal precision matrix
        n = 3

        function test_diagonal(θ)
            μ = zeros(n)
            Q = Diagonal(θ)
            gmrf = GMRF(μ, Q)
            return logdetcov(gmrf)
        end

        θ = ones(n) .* 3.0
        grad = ForwardDiff.gradient(test_diagonal, θ)

        # For diagonal Q, logdetcov = -sum(log(q_i))
        expected = -1.0 ./ θ
        @test grad ≈ expected rtol = 1.0e-6
    end

    @testset "Integration with higher-level operations" begin
        # Test that ForwardDiff works through full pipeline
        using Distributions: logpdf

        n = 5

        function full_pipeline(θ)
            # Build GMRF from parameters
            μ = θ[1:n]
            Q_diag = exp.(θ[(n + 1):(2 * n)])  # Ensure positive
            Q = Diagonal(Q_diag)

            gmrf = GMRF(μ, Q)

            # Evaluate logpdf at some point
            x = ones(n)
            return logpdf(gmrf, x)
        end

        θ0 = vcat(zeros(n), zeros(n))
        grad = ForwardDiff.gradient(full_pipeline, θ0)

        @test length(grad) == 2 * n
        @test all(isfinite.(grad))
    end
end
