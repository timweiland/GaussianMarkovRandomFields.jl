using GaussianMarkovRandomFields
using Ferrite
using SparseArrays
using LinearAlgebra
using Statistics
using ForwardDiff

@testset "BarrierModel" begin
    N = 24
    grid = generate_grid(Triangle, (N, N))     # [-1, 1]^2
    ip = Lagrange{RefTriangle, 1}()
    qr = QuadratureRule{RefTriangle}(2)
    disc = FEMDiscretization(grid, ip, qr)

    # Vertical barrier strip x ∈ [-0.1, 0.1].
    strip = [(-0.1, -2.0), (0.1, -2.0), (0.1, 2.0), (-0.1, 2.0)]

    @testset "reduces to stationary Matérn with no barrier" begin
        bm = BarrierModel(disc; barrier_cells = Int[], range_fraction = 0.1)
        mm = MaternModel(disc; smoothness = 0)
        for (τ, r) in [(1.0, 0.5), (2.0, 0.3), (0.7, 1.2)]
            Qb = sparse(precision_matrix(bm; τ = τ, range = r))
            Qm = sparse(precision_matrix(mm; τ = τ, range = r))
            @test Qb ≈ Qm rtol = 1.0e-10
        end
    end

    @testset "barrier_triangles selects centroid-in-polygon" begin
        bcells = barrier_triangles(disc, strip)
        @test !isempty(bcells)
        @test bcells == sort(unique(bcells))
        cells = grid.cells
        nodes = grid.nodes
        for cid in eachindex(cells)
            cx = mean(nodes[n].x[1] for n in cells[cid].nodes)
            @test (cid in bcells) == (-0.1 < cx < 0.1)
        end
        # Matrix-of-vertices interface gives the same answer.
        polymat = [-0.1 -2.0; 0.1 -2.0; 0.1 2.0; -0.1 2.0]
        @test barrier_triangles(disc, polymat) == bcells
    end

    @testset "precision is sparse, PD, same pattern as stationary; model interface" begin
        bcells = barrier_triangles(disc, strip)
        bm = BarrierModel(disc; barrier_cells = bcells, range_fraction = 0.1)
        mm = MaternModel(disc; smoothness = 0)
        Q = sparse(precision_matrix(bm; τ = 1.0, range = 0.5))
        Qm = sparse(precision_matrix(mm; τ = 1.0, range = 0.5))
        @test cholesky(Symmetric(Q)) isa SparseArrays.CHOLMOD.Factor
        # Barrier and stationary Matérn are both padded to the structural
        # pattern of the same product (#183), so their stored patterns — and
        # hence sparsity/cost — coincide exactly. Far from dense.
        @test nnz(Q) == nnz(Qm)
        @test nnz(Q) < 0.05 * length(Q)
        x = bm(τ = 1.0, range = 0.5)
        @test x isa GaussianMarkovRandomFields.AbstractGMRF
        @test length(bm) == ndofs(disc)
        @test length(x) == ndofs(disc)
        @test mean(bm) == zeros(ndofs(disc))
        @test Set(keys(hyperparameters(bm))) == Set([:τ, :range])
        @test model_name(bm) == :barrier
    end

    @testset "barrier blocks cross-correlation" begin
        bcells = barrier_triangles(disc, strip)
        bm = BarrierModel(disc; barrier_cells = bcells, range_fraction = 0.1)
        mm = MaternModel(disc; smoothness = 0)

        # For P1 on this grid the DOF order matches the node order.
        coords = [grid.nodes[i].x for i in 1:length(grid.nodes)]
        findnode(p) = argmin([(c[1] - p[1])^2 + (c[2] - p[2])^2 for c in coords])
        i_src = findnode((-0.45, 0.0))
        i_across = findnode((0.45, 0.0))     # dist 0.9, across the barrier
        i_within = findnode((-0.45, 0.9))    # dist 0.9, same (left) region

        function across_within(model)
            Q = sparse(precision_matrix(model; τ = 1.0, range = 0.5))
            x = model(τ = 1.0, range = 0.5)
            v = var(x)
            e = zeros(length(x))
            e[i_src] = 1.0
            Σcol = Q \ e
            c(i) = Σcol[i] / sqrt(v[i] * v[i_src])
            return c(i_across), c(i_within)
        end

        a_b, w_b = across_within(bm)
        a_s, w_s = across_within(mm)
        @test a_b < 0.2 * w_b          # barrier: across strongly suppressed vs within
        @test a_s > 0.5 * w_s          # stationary: across ≈ within (same distance)
        @test a_b < 0.2 * a_s          # barrier suppresses across vs stationary
    end

    @testset "sparsity pattern is range-invariant (#183)" begin
        bcells = barrier_triangles(disc, strip)
        bm = BarrierModel(disc; barrier_cells = bcells, range_fraction = 0.1)
        # Fill entries of Aᵀ C̃⁻¹ A whose true value is zero evaluate to exact
        # 0.0 for some ranges (then dropped by sparse scalar `*`) and to
        # roundoff for others (then kept), so the stored pattern used to be
        # range-dependent — on this mesh e.g. ranges 0.2 and 0.3 differed by
        # 6 stored entries. The precision is now padded to the precomputed
        # structural pattern, which must make all patterns identical.
        ranges = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.3, 2.0]
        Qs = [sparse(precision_matrix(bm; τ = τ, range = r)) for τ in (0.3, 1.0) for r in ranges]
        @test all(Q.colptr == Qs[1].colptr for Q in Qs)
        @test all(Q.rowval == Qs[1].rowval for Q in Qs)

        # The crash path from #183: a workspace fixes its pattern at θ_ref and
        # must accept the precision produced at every other θ.
        ws = make_workspace(bm; τ = 1.0, range = 0.3)
        for r in ranges
            x = bm(ws; τ = 1.0, range = r)
            @test x isa GaussianMarkovRandomFields.AbstractGMRF
        end
    end

    @testset "argument validation" begin
        @test_throws ArgumentError BarrierModel(disc; range_fraction = 0.0)
        @test_throws ArgumentError BarrierModel(disc; range_fraction = 1.0)
        bm = BarrierModel(disc)
        @test_throws ArgumentError precision_matrix(bm; τ = -1.0, range = 0.5)
        @test_throws ArgumentError precision_matrix(bm; τ = 1.0, range = -0.5)
    end

    @testset "differentiable through range (ForwardDiff)" begin
        bcells = barrier_triangles(disc, strip)
        bm = BarrierModel(disc; barrier_cells = bcells, range_fraction = 0.1)
        f(r) = sum(abs2, sparse(precision_matrix(bm; τ = 1.0, range = r)))
        g = ForwardDiff.derivative(f, 0.6)
        fd = (f(0.6 + 1.0e-6) - f(0.6 - 1.0e-6)) / 2.0e-6
        @test isfinite(g)
        @test g ≈ fd rtol = 1.0e-4
    end
end
