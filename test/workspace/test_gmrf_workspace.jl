using GaussianMarkovRandomFields
using GaussianMarkovRandomFields: workspace_solve, backward_solve, selinv, selinv_diag
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
end
