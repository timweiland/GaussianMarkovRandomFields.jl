using GaussianMarkovRandomFields, Ferrite, SparseArrays

@testset "FEMDiscretization" begin
    N_xy = 20
    grid = generate_grid(Triangle, (N_xy, N_xy))
    ip = Lagrange{RefTriangle, 1}()
    qr = QuadratureRule{RefTriangle}(2)

    f = FEMDiscretization(grid, ip, qr)

    @test ndim(f) == 2
    @test ndofs(f) == (N_xy + 1)^2

    X = [Tensors.Vec(0.5, 0.45), Tensors.Vec(0.67, 0.55)]
    A = evaluation_matrix(f, X)

    @test size(A) == (length(X), ndofs(f))
    @test all(A .>= 0)
    @test all(A .<= 1)
    @test sum(A, dims = 2) ≈ ones(length(X))


    node_idcs = [6, 13]
    B = node_selection_matrix(f, node_idcs)
    @test size(A) == (length(node_idcs), ndofs(f))
    @test all(B .>= 0)
    @test all(B .<= 1)
    row_idcs, col_idcs, vals = findnz(B)
    @test row_idcs == 1:length(node_idcs)
    @test all(vals .≈ 1)
end
