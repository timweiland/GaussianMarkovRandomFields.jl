using GMRFs
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
            νs = [0, 1, 2]
            grid_shape = order == 1 ? Triangle : QuadraticTriangle
            ref_shape = RefTriangle
        else
            νs = [1 // 2, 3 // 2, 5 // 2]
            grid_shape = order == 1 ? Tetrahedron : QuadraticTetrahedron
            ref_shape = RefTetrahedron
        end
        grid = generate_grid(grid_shape, Tuple(fill(N, d)))
        precisions = []
        for ν ∈ νs
            ip = Lagrange{ref_shape,order}()
            qr = QuadratureRule{ref_shape}(order + 1)
            disc = FEMDiscretization(grid, ip, qr)

            κ = rand(rng) / 2 + 0.5
            spde = MaternSPDE{d}(κ, ν)

            x = discretize(spde, disc)
            Q = to_matrix(precision_map(x))
            @test size(Q) == (ndofs(disc), ndofs(disc))
            @test cholesky(Q) isa Union{Cholesky,SparseArrays.CHOLMOD.Factor}
            push!(precisions, Q)
        end
        nnzs = [nnz(sparse(Q)) for Q in precisions]
        @test all(diff(nnzs) .> 0) # Increasing smoothness decreases sparsity
    end
end
