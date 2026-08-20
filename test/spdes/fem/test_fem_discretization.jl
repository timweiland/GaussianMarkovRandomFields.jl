using GaussianMarkovRandomFields, Ferrite, SparseArrays

@testset "FEMDiscretization" begin
    N_xy = 20
    grid = generate_grid(Triangle, (N_xy, N_xy))
    ip = Lagrange{RefTriangle, 1}()
    qr = QuadratureRule{RefTriangle}(2)

    f = FEMDiscretization(grid, ip, qr)

    @test ndim(f) == 2
    @test ndofs(f) == (N_xy + 1)^2

    X = [Vec(0.5, 0.45), Vec(0.67, 0.55)]
    A = evaluation_matrix(f, X)

    @test size(A) == (length(X), ndofs(f))
    @test all(A .>= 0)
    @test all(A .<= 1)
    @test sum(A, dims = 2) ≈ ones(length(X))

    # Test Matrix interface
    X_matrix = [0.5 0.45; 0.67 0.55]  # N×2 matrix
    A_matrix = evaluation_matrix(f, X_matrix)
    @test A_matrix ≈ A

    # Points outside the mesh cannot be located in any cell and must raise a
    # clear error rather than silently producing an empty row.
    X_oob = [Vec(0.5, 0.45), Vec(5.0, 5.0)]  # second point lies outside [-1, 1]^2
    @test_throws ArgumentError evaluation_matrix(f, X_oob)
    @test_throws ArgumentError evaluation_matrix(f, [0.5 0.45; 5.0 5.0])

    # Points exactly on cell boundaries: point location may return local
    # coordinates a roundoff outside the reference cell (does so on
    # Ferrite ≥ 1.5), and the affine weights must still land in [0, 1].
    # This mesh has grid lines at multiples of 0.1.
    X_edge = [Vec(0.5, 0.45), Vec(0.5, 0.5), Vec(-1.0, -1.0), Vec(0.3, 0.3)]
    A_edge = evaluation_matrix(f, X_edge)
    @test all(A_edge .>= 0)
    @test all(A_edge .<= 1)
    @test sum(A_edge, dims = 2) ≈ ones(length(X_edge))

    # Quadratic shape functions legitimately take negative values inside a
    # cell; the [0, 1] snap must not apply to them.
    f2 = FEMDiscretization(grid, Lagrange{RefTriangle, 2}(), QuadratureRule{RefTriangle}(4))
    A2 = evaluation_matrix(f2, [Vec(0.53, 0.415)])  # strictly inside a cell
    @test minimum(A2) < -0.01
    @test sum(A2, dims = 2) ≈ [1.0]


    node_idcs = [6, 13]
    B = node_selection_matrix(f, node_idcs)
    @test size(A) == (length(node_idcs), ndofs(f))
    @test all(B .>= 0)
    @test all(B .<= 1)
    row_idcs, col_idcs, vals = findnz(B)
    @test row_idcs == 1:length(node_idcs)
    @test all(vals .≈ 1)
end
