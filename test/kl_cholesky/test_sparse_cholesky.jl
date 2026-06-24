using GaussianMarkovRandomFields
using LinearAlgebra, SparseArrays, Random
using ForwardDiff
using ReTest

# Helper function to compute covariance approximation from L and P
function approx_cov(L, P)
    P_inv = invperm(P)
    L_inv = inv(Matrix(L))
    return (L_inv' * L_inv)[P_inv, P_inv]
end

@testset "sparse_approximate_cholesky" begin
    @testset "sparse_approximate_cholesky! basic" begin
        # Simple diagonal covariance (should be near-exact)
        Θ = Diagonal([1.0, 2.0, 3.0, 4.0])
        L = sparse([1.0 0 0 0; 1.0 1.0 0 0; 1.0 1.0 1.0 0; 1.0 1.0 1.0 1.0])  # Full lower tri pattern

        sparse_approximate_cholesky!(Θ, L)

        @test istril(L)
        Q_approx = L * L'
        @test Q_approx ≈ inv(Θ) atol = 1.0e-4
    end

    @testset "sparse_approximate_cholesky! with PermutedMatrix" begin
        rng = MersenneTwister(42)
        Θ = Symmetric(rand(rng, 4, 4) + 4I)  # Well-conditioned
        P = [3, 1, 4, 2]
        Θ_P = PermutedMatrix(Θ, P)

        # Create sparsity pattern
        L = sparse([1.0 0 0 0; 1.0 1.0 0 0; 1.0 1.0 1.0 0; 1.0 1.0 1.0 1.0])

        sparse_approximate_cholesky!(Θ_P, L)

        @test istril(L)
        @test all(diag(L) .> 0)  # Positive diagonal
    end

    @testset "sparse_approximate_cholesky wrapper" begin
        rng = MersenneTwister(123)

        # 2D grid
        X = hcat([[x, y] for x in 0:0.2:1, y in 0:0.2:1]...)
        n = size(X, 2)

        # Simple covariance matrix
        K = zeros(n, n)
        for i in 1:n, j in 1:n
            K[i, j] = exp(-norm(X[:, i] - X[:, j])^2 / 0.5)
        end

        # Test with different ρ values
        for ρ in [1.5, 2.0, 2.5]
            L, P = sparse_approximate_cholesky(K, X; ρ = ρ)

            @test issparse(L)
            @test istril(L)
            @test isperm(P)
            @test size(L) == (n, n)

            # Larger ρ should give denser approximation
            if ρ == 2.5
                @test nnz(L) > nnz(sparse_approximate_cholesky(K, X; ρ = 1.5)[1])
            end
        end
    end

    @testset "Approximation quality" begin
        rng = MersenneTwister(456)

        # Small 1D grid
        X = reshape(0.0:0.1:1.0, 1, :)
        n = size(X, 2)

        # Matern-like covariance
        K = zeros(n, n)
        for i in 1:n, j in 1:n
            r = norm(X[:, i] - X[:, j])
            K[i, j] = (1 + sqrt(3) * r / 0.3) * exp(-sqrt(3) * r / 0.3)
        end

        L, P = sparse_approximate_cholesky(K, X; ρ = 4.0)
        K_approx = approx_cov(L, P)

        # Relative error should be reasonable
        rel_error = norm(K_approx - K) / norm(K)
        @test rel_error < 0.5  # Loose bound for approximation
    end

    @testset "Supernodal vs non-supernodal" begin
        rng = MersenneTwister(789)

        # Small test problem
        X = hcat([[x, y] for x in 0:0.3:1, y in 0:0.3:1]...)
        n = size(X, 2)

        K = zeros(n, n)
        for i in 1:n, j in 1:n
            K[i, j] = exp(-norm(X[:, i] - X[:, j])^2 / 0.4)
        end

        # Test supernodal (default)
        L_super, P_super = sparse_approximate_cholesky(K, X; ρ = 2.5, λ = 1.5)
        K_approx_super = approx_cov(L_super, P_super)
        rel_error_super = norm(K_approx_super - K) / norm(K)

        # Test non-supernodal
        L_nonsup, P_nonsup = sparse_approximate_cholesky(K, X; ρ = 2.5, λ = nothing)
        K_approx_nonsup = approx_cov(L_nonsup, P_nonsup)
        rel_error_nonsup = norm(K_approx_nonsup - K) / norm(K)

        # Both should produce acceptable approximations
        @test rel_error_super < 0.5
        @test rel_error_nonsup < 0.5
    end

    @testset "Edge cases" begin
        # 2x2 matrix
        X_small = [0.0 1.0; 0.0 0.0]
        K_small = [1.0 0.5; 0.5 1.0]
        L, P = sparse_approximate_cholesky(K_small, X_small; ρ = 2.0)
        @test size(L) == (2, 2)
        @test isperm(P)
    end

    @testset "ForwardDiff: AD through the factor w.r.t. covariance values" begin
        # The factorisation is a fixed-pattern sweep of small dense block-Choleskys + unit-RHS
        # solves, smooth in Θ's entries; with eltype-generic buffers a ForwardDiff.Dual flows
        # Θ -> L. (The ordering / sparsity pattern come from the points X, which are AD-invariant.)
        X = hcat([[x, y] for x in 0:0.25:1, y in 0:0.25:1]...)
        n = size(X, 2)
        kern(ℓ) = [exp(-norm(X[:, i] - X[:, j])^2 / (2ℓ^2)) for i in 1:n, j in 1:n] + 1.0e-4 * I
        f(ℓ; λ) = sum(abs2, nonzeros(sparse_approximate_cholesky(kern(ℓ), X; ρ = 2.0, λ = λ)[1]))

        ℓ0 = 0.4
        # Gradient matches central differences on both the supernodal and column-by-column paths.
        for λ in (nothing, 1.5)
            d_ad = ForwardDiff.derivative(ℓ -> f(ℓ; λ = λ), ℓ0)
            d_fd = (f(ℓ0 + 1.0e-6; λ = λ) - f(ℓ0 - 1.0e-6; λ = λ)) / 2.0e-6
            @test d_ad ≈ d_fd rtol = 1.0e-5
        end

        # The Dual-eltype factor has the same pattern and the same primal values as the Float64
        # factor (the partials ride along without perturbing the value).
        L_d, P_d = sparse_approximate_cholesky(kern(ForwardDiff.Dual(ℓ0, 1.0)), X; ρ = 2.0)
        L_f, P_f = sparse_approximate_cholesky(kern(ℓ0), X; ρ = 2.0)
        @test eltype(L_d) <: ForwardDiff.Dual
        @test P_d == P_f
        @test L_d.colptr == L_f.colptr && rowvals(L_d) == rowvals(L_f)
        @test ForwardDiff.value.(nonzeros(L_d)) ≈ nonzeros(L_f) rtol = 1.0e-10
        # Float64 inputs are untouched: promote_type collapses to Float64 (the LAPACK path).
        @test eltype(L_f) == Float64
    end

    @testset "ForwardDiff: in-place sparse_approximate_cholesky! with a Dual factor" begin
        Θ = [2.0 0.5 0.1; 0.5 2.0 0.3; 0.1 0.3 2.0]   # SPD
        pat = sparse([1.0 0 0; 1.0 1.0 0; 1.0 1.0 1.0])
        function g(s)
            L = SparseMatrixCSC{typeof(s), Int}(pat)
            sparse_approximate_cholesky!(s .* Θ, L)
            return sum(abs2, nonzeros(L))
        end
        d_ad = ForwardDiff.derivative(g, 1.0)
        d_fd = (g(1.0 + 1.0e-6) - g(1.0 - 1.0e-6)) / 2.0e-6
        @test d_ad ≈ d_fd rtol = 1.0e-5
    end
end
