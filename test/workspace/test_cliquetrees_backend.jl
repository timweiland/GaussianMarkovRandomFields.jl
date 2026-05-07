using GaussianMarkovRandomFields
using GaussianMarkovRandomFields: workspace_solve, backward_solve, selinv, selinv_diag
using LinearAlgebra
using SparseArrays
using Random

@testset "CliqueTreesBackend" begin
    n = 20
    Q = _make_test_precision(n)
    Q_dense = Matrix(Q)
    Q_inv = inv(Q_dense)

    @testset "Construction" begin
        ws = GMRFWorkspace(Q, CliqueTreesBackend)
        @test dimension(ws) == n
        @test ws.backend isa CliqueTreesBackend
    end

    @testset "Solve" begin
        ws = GMRFWorkspace(Q, CliqueTreesBackend)
        b = randn(n)
        x = workspace_solve(ws, b)
        @test x ≈ Q_dense \ b rtol = 1.0e-10
    end

    @testset "Log determinant" begin
        ws = GMRFWorkspace(Q, CliqueTreesBackend)
        @test logdet(ws) ≈ logdet(Q_dense) rtol = 1.0e-10
    end

    @testset "Selected inverse diagonal" begin
        ws = GMRFWorkspace(Q, CliqueTreesBackend)
        d = selinv_diag(ws)
        @test d ≈ diag(Q_inv) rtol = 1.0e-8
    end

    @testset "Full selected inverse" begin
        ws = GMRFWorkspace(Q, CliqueTreesBackend)
        Qinv_sel = selinv(ws)
        rows = rowvals(Qinv_sel)
        vals = nonzeros(Qinv_sel)
        for col in 1:n
            for idx in nzrange(Qinv_sel, col)
                row = rows[idx]
                @test vals[idx] ≈ Q_inv[row, col] rtol = 1.0e-6
            end
        end
    end

    @testset "Update precision and refactorize" begin
        ws = GMRFWorkspace(Q, CliqueTreesBackend)

        Q2 = copy(Q)
        Q2.nzval .*= 2.0
        Q2_dense = Matrix(Q2)
        Q2_inv = inv(Q2_dense)

        update_precision!(ws, Q2)

        b = randn(n)
        @test workspace_solve(ws, b) ≈ Q2_dense \ b rtol = 1.0e-10
        @test logdet(ws) ≈ logdet(Q2_dense) rtol = 1.0e-10
        @test selinv_diag(ws) ≈ diag(Q2_inv) rtol = 1.0e-8
    end

    @testset "WorkspaceGMRF with CliqueTreesBackend" begin
        ws = GMRFWorkspace(Q, CliqueTreesBackend)
        μ = randn(n)
        wg = WorkspaceGMRF(μ, Q, ws)

        ref = GMRF(μ, Q)
        z = randn(n)
        @test logpdf(wg, z) ≈ logpdf(ref, z) rtol = 1.0e-8
        @test var(wg) ≈ var(ref) rtol = 1.0e-8
    end

    @testset "Thread-safe parallel execution" begin
        n_tasks = 20
        results = Vector{Float64}(undef, n_tasks)

        pool = Channel{GMRFWorkspace}(4)
        for _ in 1:4
            put!(pool, GMRFWorkspace(copy(Q), CliqueTreesBackend))
        end

        @sync for i in 1:n_tasks
            Threads.@spawn begin
                ws = take!(pool)
                try
                    scale = Float64(i)
                    Q_scaled = copy(Q)
                    Q_scaled.nzval .*= scale
                    update_precision!(ws, Q_scaled)
                    b = ones(n)
                    x = workspace_solve(ws, b)
                    results[i] = norm(x)
                finally
                    put!(pool, ws)
                end
            end
        end

        for i in 1:n_tasks
            Q_scaled_dense = Matrix(Q) * i
            expected = norm(Q_scaled_dense \ ones(n))
            @test results[i] ≈ expected rtol = 1.0e-10
        end
    end
end
