using GaussianMarkovRandomFields
using GaussianMarkovRandomFields: workspace_solve, backward_solve, selinv, selinv_diag, selinv_dot,
    selinv_extract_at
using LinearAlgebra
using SparseArrays
using Random

function _make_test_precision(n; rng = Random.MersenneTwister(42))
    # Create a sparse SPD matrix with known structure
    Q = sprand(rng, n, n, 0.3)
    Q = Q * Q' + n * I
    return sparse(Q)
end

@testset "GMRFWorkspace" begin
    n = 20
    Q = _make_test_precision(n)
    Q_dense = Matrix(Q)
    Q_inv = inv(Q_dense)

    @testset "Construction" begin
        ws = GMRFWorkspace(Q)
        @test dimension(ws) == n
    end

    @testset "Solve" begin
        ws = GMRFWorkspace(Q)
        b = randn(n)
        x = workspace_solve(ws, b)
        @test x ≈ Q_dense \ b
    end

    @testset "Log determinant" begin
        ws = GMRFWorkspace(Q)
        @test logdet(ws) ≈ logdet(Q_dense) rtol = 1.0e-10
    end

    @testset "Selected inverse diagonal" begin
        ws = GMRFWorkspace(Q)
        d = selinv_diag(ws)
        @test d ≈ diag(Q_inv) rtol = 1.0e-8
    end

    @testset "Full selected inverse" begin
        ws = GMRFWorkspace(Q)
        Qinv_sel = selinv(ws)
        # selinv returns Q⁻¹ at nonzero positions of the Cholesky factor pattern
        # Check that values at nonzero positions match the full inverse
        rows = rowvals(Qinv_sel)
        vals = nonzeros(Qinv_sel)
        for col in 1:n
            for idx in nzrange(Qinv_sel, col)
                row = rows[idx]
                @test vals[idx] ≈ Q_inv[row, col] rtol = 1.0e-6
            end
        end
    end

    @testset "selinv_dot" begin
        ws = GMRFWorkspace(Q)
        # tr(Q⁻¹ Q) = tr(I) = n  (selinv has all of Q's pattern, so the contraction is exact)
        @test selinv_dot(ws, Q) ≈ n rtol = 1.0e-8
        # Matches the materialized-selinv dot for an arbitrary matrix on Q's pattern
        B = copy(Q)
        B.nzval .= randn(length(B.nzval))
        @test selinv_dot(ws, B) ≈ dot(selinv(ws), B) rtol = 1.0e-10
    end

    @testset "selinv_extract_at" begin
        # Σ read at a subset pattern must equal the materialized selinv masked to that
        # pattern (bit-identical), on simplicial (small) and supernodal (larger) factors.
        for Qt in (Q, _make_test_precision(400))
            ws = GMRFWorkspace(Qt)
            nt = size(Qt, 1)
            B = copy(Qt)   # Qt's pattern ⊆ the Cholesky fill, so all entries are stored
            Σe = selinv_extract_at(ws, B)
            Σf = selinv(ws)
            @test Σe.colptr == B.colptr
            @test rowvals(Σe) == rowvals(B)
            rv = rowvals(B)
            @test all(Σe[rv[t], j] == Σf[rv[t], j] for j in 1:nt for t in nzrange(B, j))
        end
    end

    @testset "Backward solve" begin
        ws = GMRFWorkspace(Q)
        z = randn(n)
        x = backward_solve(ws, z)
        # backward_solve(ws, z) maps a standard-normal z to a sample from N(0, Q^{-1}),
        # so the empirical sample variance should converge to diag(Q^{-1}).
        rng = Random.MersenneTwister(123)
        n_samples = 50000
        samples = zeros(n, n_samples)
        for i in 1:n_samples
            z_i = randn(rng, n)
            samples[:, i] = backward_solve(ws, z_i)
        end
        empirical_cov = (samples * samples') / n_samples
        @test diag(empirical_cov) ≈ diag(Q_inv) rtol = 0.1
    end

    @testset "Update precision with new values" begin
        ws = GMRFWorkspace(Q)

        # Scale Q — preserves pattern and SPD
        Q2 = copy(Q)
        Q2.nzval .*= 2.0

        Q2_dense = Matrix(Q2)
        Q2_inv = inv(Q2_dense)

        update_precision!(ws, Q2)

        b = randn(n)
        @test workspace_solve(ws, b) ≈ Q2_dense \ b
        @test logdet(ws) ≈ logdet(Q2_dense) rtol = 1.0e-10
        @test selinv_diag(ws) ≈ diag(Q2_inv) rtol = 1.0e-8
    end

    @testset "Update precision values directly" begin
        ws = GMRFWorkspace(Q)

        new_nzval = Q.nzval .* 3.0
        update_precision_values!(ws, new_nzval)

        Q3 = copy(Q)
        Q3.nzval .= new_nzval
        Q3_dense = Matrix(Q3)

        b = randn(n)
        @test workspace_solve(ws, b) ≈ Q3_dense \ b
        @test logdet(ws) ≈ logdet(Q3_dense) rtol = 1.0e-10
    end

    @testset "Persistent factorization buffer across many updates" begin
        # The CHOLMOD backend reuses a persistent CHOLMOD.Sparse, refreshing its
        # values in place on each refactorization rather than rebuilding it. Drive
        # a sequence of distinct updates on one workspace and check every
        # solve/logdet/selinv against a freshly built dense reference — this
        # catches any stale-value carryover in the reused value buffer.
        ws = GMRFWorkspace(Q)
        rng = Random.MersenneTwister(7)
        for k in 1:6
            Qk = copy(Q)
            Qk.nzval .*= 0.5 + k          # vary the overall scale...
            for i in 1:n
                Qk[i, i] += 0.3 * k * i / n   # ...and perturb the diagonal non-uniformly
            end

            update_precision!(ws, Qk)

            Qk_dense = Matrix(Qk)
            b = randn(rng, n)
            @test workspace_solve(ws, b) ≈ Qk_dense \ b
            @test logdet(ws) ≈ logdet(Qk_dense) rtol = 1.0e-10
            @test selinv_diag(ws) ≈ diag(inv(Qk_dense)) rtol = 1.0e-8
        end
    end

    @testset "Pattern mismatch error" begin
        ws = GMRFWorkspace(Q)
        Q_different = sprand(n, n, 0.1)
        Q_different = Q_different * Q_different' + n * I
        Q_different = sparse(Q_different)
        # This should have a different sparsity pattern (with high probability)
        @test_throws ArgumentError update_precision!(ws, Q_different)
    end

    @testset "Lazy invalidation" begin
        ws = GMRFWorkspace(Q)

        # First access computes selinv
        d1 = selinv_diag(ws)
        @test d1 ≈ diag(Q_inv) rtol = 1.0e-8

        # Update values
        Q2 = copy(Q)
        Q2.nzval .*= 2.0
        update_precision!(ws, Q2)

        # selinv should be recomputed
        Q2_inv = inv(Matrix(Q2))
        d2 = selinv_diag(ws)
        @test d2 ≈ diag(Q2_inv) rtol = 1.0e-8
        @test !(d1 ≈ d2)  # Should be different

        # logdet should also be recomputed
        @test logdet(ws) ≈ logdet(Matrix(Q2)) rtol = 1.0e-10
    end

    @testset "Cached results are reused" begin
        ws = GMRFWorkspace(Q)

        # Compute selinv twice — second call should return cached result
        s1 = selinv(ws)
        s2 = selinv(ws)
        @test s1 === s2  # Same object (cached)

        d1 = selinv_diag(ws)
        d2 = selinv_diag(ws)
        @test d1 === d2  # Same object (cached)
    end

    @testset "Selinv lazy materialization" begin
        # Issue #144: the CHOLMOD backend materializes the selected inverse
        # lazily. Callers needing only the diagonal must not build the full
        # sparse matrix, and an already-built full selinv should be reused for
        # the diagonal rather than recomputed.

        # Diagonal-only access never materializes the full sparse selected inverse.
        ws = GMRFWorkspace(Q)
        d = selinv_diag(ws)
        @test d ≈ diag(Q_inv) rtol = 1.0e-8
        @test ws.backend.selinv_cache === nothing

        # When the full selinv is already built, the diagonal is taken from it.
        ws2 = GMRFWorkspace(Q)
        S = selinv(ws2)
        @test ws2.backend.selinv_cache !== nothing
        @test selinv_diag(ws2) == diag(S)

        # The diagonal-only path and the full path agree bit-for-bit.
        @test selinv_diag(GMRFWorkspace(Q)) == diag(selinv(GMRFWorkspace(Q)))
    end
end
