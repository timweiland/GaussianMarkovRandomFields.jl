using GaussianMarkovRandomFields
using Random
using Ferrite
using SparseArrays
using LinearAlgebra

@testset "MaternSPDE" begin
    rng = MersenneTwister(5930238)
    N = 20
    @testset "$d-dimensional, element order $order" for d ∈ [1, 2, 3], order ∈ [1, 2]
        if d == 3 && order == 2
            continue # TODO: generate_grid does not support QuadraticTetrahedron (yet?)
        end
        if d == 1
            νs = [1 // 2, 3 // 2, 5 // 2]
            grid_shape = order == 1 ? Line : QuadraticLine
            ref_shape = RefLine
        elseif d == 2
            νs = [1, 2, 3]
            grid_shape = order == 1 ? Triangle : QuadraticTriangle
            ref_shape = RefTriangle
        else
            νs = [1 // 2, 3 // 2, 5 // 2]
            grid_shape = order == 1 ? Tetrahedron : QuadraticTetrahedron
            ref_shape = RefTetrahedron
        end
        grid = generate_grid(grid_shape, Tuple(fill(N, d)))
        precisions = []
        for (smoothness_idx, ν) ∈ enumerate(νs)
            ip = Lagrange{ref_shape,order}()
            qr = QuadratureRule{ref_shape}(order + 1)
            disc = FEMDiscretization(grid, ip, qr)

            κ = rand(rng) / 2 + 0.5
            spde = MaternSPDE{d}(κ = κ, ν = ν)
            spde_smoothness = MaternSPDE{d}(κ = κ, smoothness = smoothness_idx - 1)
            @test spde_smoothness.ν ≈ spde.ν

            x = GaussianMarkovRandomFields.discretize(spde, disc)
            Q = to_matrix(precision_map(x))
            @test size(Q) == (ndofs(disc), ndofs(disc))
            @test cholesky(Q) isa Union{Cholesky,SparseArrays.CHOLMOD.Factor}
            push!(precisions, Q)

            # Test range parameter
            spde_high_range = MaternSPDE{d}(range = 1.0, smoothness = smoothness_idx - 1)
            spde_low_range = MaternSPDE{d}(range = 0.1, smoothness = smoothness_idx - 1)

            x_high_range = GaussianMarkovRandomFields.discretize(spde_high_range, disc)
            x_low_range = GaussianMarkovRandomFields.discretize(spde_low_range, disc)

            A_eval = evaluation_matrix(disc, [Tensors.Vec(fill(0.0, d)...)])
            y = [1.0]

            x_high_cond = condition_on_observations(x_high_range, A_eval, 1e8, y)
            x_low_cond = condition_on_observations(x_low_range, A_eval, 1e8, y)

            A_test = evaluation_matrix(disc, [Tensors.Vec(fill(0.5, d)...)])
            @test A_test * mean(x_high_cond) > A_test * mean(x_low_cond) .+ 0.1
        end
        nnzs = [nnz(sparse(Q)) for Q in precisions]
        @test all(diff(nnzs) .> 0) # Increasing smoothness decreases sparsity
    end

    @testset "Boundary conditions" begin
        grid = generate_grid(Line, (50,))
        ip = Lagrange{RefLine,1}()
        qr = QuadratureRule{RefLine}(2)
        pbc = _get_periodic_constraint(grid)
        left_val, right_val = 2.0, -1.5
        dbc = _get_dirichlet_constraint(grid, left_val, right_val)
        disc_pbc = FEMDiscretization(grid, ip, qr, ((:u, nothing),), [(pbc, 1e-4)])
        disc_dbc = FEMDiscretization(grid, ip, qr, ((:u, nothing),), [(dbc, 1e-4)])

        for smoothness ∈ [1, 3]
            spde = MaternSPDE{1}(range = 0.3, smoothness = smoothness, σ² = 0.3)
            x = GaussianMarkovRandomFields.discretize(spde, disc_pbc)

            @test (mean(x)[1] ≈ 0.0) && (mean(x)[end] ≈ 0.0)

            for i = 1:3
                samp = rand(rng, x)
                @test samp[1] ≈ samp[end] atol = 1e-3
            end
        end

        for smoothness ∈ [0, 1, 2]
            spde = MaternSPDE{1}(range = 0.3, smoothness = smoothness, σ² = 0.3)
            x = GaussianMarkovRandomFields.discretize(spde, disc_dbc)

            @test (mean(x)[1] ≈ left_val) && (mean(x)[end] ≈ right_val)

            for i = 1:3
                samp = rand(rng, x)
                @test samp[1] ≈ left_val atol = 1e-3
                @test samp[end] ≈ right_val atol = 1e-3
            end

            @test std(x)[1] ≈ 1e-4
            @test std(x)[end] ≈ 1e-4
        end
    end
end
