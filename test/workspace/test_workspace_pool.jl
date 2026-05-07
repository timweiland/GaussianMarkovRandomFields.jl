using GaussianMarkovRandomFields
using GaussianMarkovRandomFields: workspace_solve, checkout, checkin, with_workspace
using LinearAlgebra
using SparseArrays

@testset "WorkspacePool" begin
    n = 20
    Q = _make_test_precision(n)

    @testset "Construction" begin
        pool = WorkspacePool(Q; size = 4)
        @test pool.size == 4
    end

    @testset "Single-threaded checkout/checkin" begin
        pool = WorkspacePool(Q; size = 2)

        ws1 = checkout(pool)
        @test ws1 isa GMRFWorkspace
        @test dimension(ws1) == n

        ws2 = checkout(pool)
        @test ws2 isa GMRFWorkspace
        @test ws1 !== ws2  # different objects

        checkin(pool, ws1)
        checkin(pool, ws2)

        # Can check out again after returning
        ws3 = checkout(pool)
        @test ws3 isa GMRFWorkspace
        checkin(pool, ws3)
    end

    @testset "with_workspace RAII pattern" begin
        pool = WorkspacePool(Q; size = 2)

        result = with_workspace(pool) do ws
            b = randn(n)
            x = workspace_solve(ws, b)
            return norm(x)
        end
        @test result > 0
        @test isfinite(result)
    end

    @testset "with_workspace returns on exception" begin
        pool = WorkspacePool(Q; size = 1)

        # Workspace should be returned even if the function throws
        try
            with_workspace(pool) do ws
                error("intentional error")
            end
        catch
        end

        # Pool should still have the workspace available
        ws = checkout(pool)
        @test ws isa GMRFWorkspace
        checkin(pool, ws)
    end

    @testset "Independent workspaces produce correct results" begin
        pool = WorkspacePool(Q; size = 3)
        Q_dense = Matrix(Q)

        # Each workspace should independently produce correct solves
        results = Vector{Vector{Float64}}(undef, 3)
        b = randn(n)

        for i in 1:3
            ws = checkout(pool)
            # Scale Q differently for each workspace
            Q_scaled = copy(Q)
            Q_scaled.nzval .*= Float64(i)
            update_precision!(ws, Q_scaled)
            results[i] = workspace_solve(ws, b)
            checkin(pool, ws)
        end

        for i in 1:3
            Q_scaled_dense = Matrix(Q) * i
            @test results[i] ≈ Q_scaled_dense \ b
        end
    end

    @testset "Parallel correctness" begin
        pool = WorkspacePool(Q; size = Threads.nthreads())
        Q_dense = Matrix(Q)
        n_tasks = 20
        results = Vector{Float64}(undef, n_tasks)

        # Each task: checkout, solve with scaled Q, checkin
        @sync for i in 1:n_tasks
            Threads.@spawn begin
                with_workspace(pool) do ws
                    scale = Float64(i)
                    Q_scaled = copy(Q)
                    Q_scaled.nzval .*= scale
                    update_precision!(ws, Q_scaled)
                    b = ones(n)
                    x = workspace_solve(ws, b)
                    results[i] = norm(x)
                end
            end
        end

        # Verify each result matches serial computation
        for i in 1:n_tasks
            Q_scaled_dense = Matrix(Q) * i
            expected = norm(Q_scaled_dense \ ones(n))
            @test results[i] ≈ expected rtol = 1.0e-10
        end
    end

    @testset "Pool from LatentModel" begin
        model = AR1Model(n)
        pool = WorkspacePool(model; size = 2, τ = 1.0, ρ = 0.5)
        @test pool.size == 2

        ws = checkout(pool)
        @test dimension(ws) == n
        checkin(pool, ws)
    end
end
