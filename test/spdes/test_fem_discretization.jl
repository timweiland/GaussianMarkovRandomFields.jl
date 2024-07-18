using GMRFs
using Ferrite

@testset "FEMDiscretization" begin
    N_xy = 20
    grid = generate_grid(Triangle, (N_xy, N_xy))
    ip = Lagrange{2,RefTetrahedron,1}()
    qr = QuadratureRule{2,RefTetrahedron}(2)

    f = FEMDiscretization(grid, ip, qr)

    @test ndim(f) == 2
    @test ndofs(f) == (N_xy + 1)^2

    X = [Tensors.Vec(0.5, 0.45), Tensors.Vec(0.67, 0.55)]
    A = evaluation_matrix(f, X)

    @test size(A) == (length(X), ndofs(f))
    @test all(A .>= 0)
    @test all(A .<= 1)
    @test sum(A, dims = 2) ≈ ones(length(X))
end
