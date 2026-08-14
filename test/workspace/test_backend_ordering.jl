using Test
using GaussianMarkovRandomFields
using LinearAlgebra, SparseArrays
import CliqueTrees

@testset "Backend ordering override" begin
    # 2D grid-graph precision + a dense border row/col (global effect):
    # structured enough that orderings differ, small enough to be instant.
    nx = 12
    n = nx * nx
    A1 = spdiagm(0 => 2.0 * ones(nx), 1 => -ones(nx - 1), -1 => -ones(nx - 1))
    Qgrid = kron(sparse(1.0I, nx, nx), A1) + kron(A1, sparse(1.0I, nx, nx)) +
        0.1 * sparse(1.0I, n, n)
    h = 0.01 * ones(n)
    Q = [Qgrid sparse(reshape(h, n, 1)); sparse(reshape(h, 1, n)) sparse([1], [1], 2.0, 1, 1)]
    N = n + 1
    rhs = randn(N)

    ws0 = GMRFWorkspace(Q)
    x0 = GaussianMarkovRandomFields.backend_solve(ws0.backend, rhs)
    ld0 = GaussianMarkovRandomFields.compute_logdet(ws0.backend)
    GaussianMarkovRandomFields.ensure_selinv!(ws0)
    d0 = GaussianMarkovRandomFields.get_selinv_diag(ws0.backend)

    @testset "explicit permutation vector" begin
        perm = collect(N:-1:1)
        ws = GMRFWorkspace(Q; ordering = perm)
        @test GaussianMarkovRandomFields.backend_solve(ws.backend, rhs) ≈ x0 rtol = 1.0e-10
        @test GaussianMarkovRandomFields.compute_logdet(ws.backend) ≈ ld0 rtol = 1.0e-10
        GaussianMarkovRandomFields.ensure_selinv!(ws)
        @test GaussianMarkovRandomFields.get_selinv_diag(ws.backend) ≈ d0 rtol = 1.0e-8
    end

    @testset "CliqueTrees elimination algorithm" begin
        ws = GMRFWorkspace(Q; ordering = CliqueTrees.MMD())
        @test GaussianMarkovRandomFields.backend_solve(ws.backend, rhs) ≈ x0 rtol = 1.0e-10
        @test GaussianMarkovRandomFields.compute_logdet(ws.backend) ≈ ld0 rtol = 1.0e-10
        GaussianMarkovRandomFields.ensure_selinv!(ws)
        @test GaussianMarkovRandomFields.get_selinv_diag(ws.backend) ≈ d0 rtol = 1.0e-8
    end

    @testset "PinDenseColumns pins the border last" begin
        w = PinDenseColumns(CliqueTrees.MMD())
        p = ordering_permutation(Q, w)
        @test isperm(p)
        @test p[end] == N                   # the dense border column
        ws = GMRFWorkspace(Q; ordering = w)
        @test GaussianMarkovRandomFields.backend_solve(ws.backend, rhs) ≈ x0 rtol = 1.0e-10
        # no dense columns → transparent passthrough to the inner algorithm
        p2 = ordering_permutation(Qgrid, w)
        @test isperm(p2) && length(p2) == n
    end

    @testset "pool shares one resolved ordering" begin
        pool = GaussianMarkovRandomFields.WorkspacePool(Q; size = 2, ordering = CliqueTrees.MMD())
        GaussianMarkovRandomFields.with_workspace(pool) do ws
            @test GaussianMarkovRandomFields.backend_solve(ws.backend, rhs) ≈ x0 rtol = 1.0e-10
        end
    end

    @testset "refactorization keeps the custom symbolic" begin
        ws = GMRFWorkspace(Q; ordering = CliqueTrees.MMD())
        Q2 = 2.0 * Q
        GaussianMarkovRandomFields.update_precision!(ws, Q2)
        GaussianMarkovRandomFields.ensure_numeric!(ws)
        @test GaussianMarkovRandomFields.compute_logdet(ws.backend) ≈ ld0 + N * log(2.0) rtol = 1.0e-9
    end
end
