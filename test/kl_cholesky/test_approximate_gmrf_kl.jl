using GaussianMarkovRandomFields
using LinearAlgebra, SparseArrays, Random
using Distributions
using ReTest

@testset "approximate_gmrf_kl" begin
    @testset "Basic GMRF construction" begin
        rng = MersenneTwister(42)

        # Small 2D grid
        X = hcat([[x, y] for x in 0:0.2:1, y in 0:0.2:1]...)
        n = size(X, 2)

        # Simple kernel matrix
        K = zeros(n, n)
        for i in 1:n, j in 1:n
            K[i, j] = exp(-norm(X[:, i] - X[:, j])^2 / 0.5)
        end

        gmrf = approximate_gmrf_kl(K, X; ρ = 2.0)

        @test gmrf isa GMRF
        @test length(gmrf) == n
        @test mean(gmrf) == zeros(n)
        @test issparse(precision_matrix(gmrf))
    end

    @testset "Precision matrix properties" begin
        rng = MersenneTwister(123)

        X = hcat([[x] for x in 0:0.1:1]...)  # 1D grid
        n = size(X, 2)

        K = zeros(n, n)
        for i in 1:n, j in 1:n
            r = abs(X[1, i] - X[1, j])
            K[i, j] = exp(-r^2 / 0.2)
        end

        gmrf = approximate_gmrf_kl(K, X; ρ = 2.0)
        Q = precision_matrix(gmrf)

        @test issparse(Q)
        @test issymmetric(Q)
        @test size(Q) == (n, n)
        @test all(diag(Q) .> 0)  # Positive diagonal
    end

    @testset "Sampling" begin
        rng = MersenneTwister(456)

        X = rand(rng, 2, 20)
        n = size(X, 2)

        # Simple isotropic kernel
        K = exp.(-0.5 * [norm(X[:, i] - X[:, j])^2 for i in 1:n, j in 1:n])

        gmrf = approximate_gmrf_kl(K, X; ρ = 2.0)

        # Should be able to sample
        sample = rand(rng, gmrf)
        @test length(sample) == n
        @test sample isa Vector{Float64}

        # Multiple samples
        samples = [rand(rng, gmrf) for _ in 1:100]
        @test length(samples) == 100
    end

    @testset "Parameter ρ effects" begin
        rng = MersenneTwister(789)

        X = hcat([[x, y] for x in 0:0.3:1, y in 0:0.3:1]...)
        n = size(X, 2)

        K = exp.(-[norm(X[:, i] - X[:, j]) for i in 1:n, j in 1:n])

        gmrf_sparse = approximate_gmrf_kl(K, X; ρ = 1.5)
        gmrf_dense = approximate_gmrf_kl(K, X; ρ = 3.0)

        Q_sparse = precision_matrix(gmrf_sparse)
        Q_dense = precision_matrix(gmrf_dense)

        # Larger ρ should give denser precision matrix
        @test nnz(Q_dense) > nnz(Q_sparse)
    end

    @testset "Integration with conditioning" begin
        rng = MersenneTwister(321)

        X = hcat([[x] for x in 0:0.2:1]...)
        n = size(X, 2)

        K = [exp(-abs(X[1, i] - X[1, j]) / 0.3) for i in 1:n, j in 1:n]

        gmrf = approximate_gmrf_kl(K, X; ρ = 2.0)

        # Condition on a few observations
        obs_indices = [1, div(n, 2), n]
        y_obs = [1.0, 0.0, -1.0]

        A = sparse(1:3, obs_indices, ones(3), 3, n)
        Q_ϵ = sparse(Diagonal(fill(100.0, 3)))

        posterior = linear_condition(gmrf; A = A, Q_ϵ = Q_ϵ, y = y_obs)

        @test posterior isa GMRF
        @test length(posterior) == n

        # Posterior mean should be close to observations at observed locations
        μ_post = mean(posterior)
        @test μ_post[obs_indices] ≈ y_obs atol = 0.1
    end

    @testset "Supernodal vs non-supernodal" begin
        rng = MersenneTwister(654)

        X = hcat([[x, y] for x in 0:0.25:1, y in 0:0.25:1]...)
        n = size(X, 2)

        K = exp.(-0.5 * [norm(X[:, i] - X[:, j])^2 for i in 1:n, j in 1:n])

        # Test supernodal (default)
        gmrf_super = approximate_gmrf_kl(K, X; ρ = 2.0, λ = 1.5)
        @test gmrf_super isa GMRF
        @test length(gmrf_super) == n
        @test issparse(precision_matrix(gmrf_super))

        # Test non-supernodal
        gmrf_nonsup = approximate_gmrf_kl(K, X; ρ = 2.0, λ = nothing)
        @test gmrf_nonsup isa GMRF
        @test length(gmrf_nonsup) == n
        @test issparse(precision_matrix(gmrf_nonsup))
    end

    @testset "Edge cases" begin
        # Small problem
        X_small = [0.0 1.0; 0.0 0.0]
        K_small = [1.0 0.8; 0.8 1.0]

        gmrf_small = approximate_gmrf_kl(K_small, X_small; ρ = 2.0)
        @test length(gmrf_small) == 2
        @test gmrf_small isa GMRF
    end
end
