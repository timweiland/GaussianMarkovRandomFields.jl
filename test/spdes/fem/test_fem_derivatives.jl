using GaussianMarkovRandomFields, Ferrite, SparseArrays, Tensors

@testset "FEM Derivatives" begin
    N_xy = 20

    @testset "Order $d" for d ∈ [1, 2]
        element = (d == 1) ? Triangle : QuadraticTriangle
        grid = generate_grid(element, (N_xy, N_xy))
        ip = Lagrange{RefTriangle,d}()
        qr = QuadratureRule{RefTriangle}(3)
        f = FEMDiscretization(grid, ip, qr)

        X = [Tensors.Vec(0.5, 0.45), Tensors.Vec(0.67, 0.55)]
        peh = PointEvalHandler(f.grid, X)
        cc = CellCache(f.dof_handler)

        @testset "Local shape function derivatives" begin
            for i = 1:getnbasefunctions(ip), j = 1:length(X)
                ξ = peh.local_coords[j]
                ∇ϕᵢ = GaussianMarkovRandomFields.shape_gradient_local(f, i, ξ)
                Hϕᵢ = GaussianMarkovRandomFields.shape_hessian_local(f, i, ξ)
                @test size(∇ϕᵢ) == (ndim(f),)
                @test size(Hϕᵢ) == (ndim(f), ndim(f))
                if d == 1
                    @test all(Hϕᵢ .≈ 0)
                end
            end
        end

        @testset "Global shape function derivatives" begin
            for i = 1:getnbasefunctions(ip), j = 1:length(X)
                Ferrite.reinit!(cc, peh.cells[j])
                ξ = peh.local_coords[j]
                dof_coords = getcoordinates(cc)
                ∇ϕᵢ = GaussianMarkovRandomFields.shape_gradient_global(f, dof_coords, i, ξ)
                Hϕᵢ = GaussianMarkovRandomFields.shape_hessian_global(f, dof_coords, i, ξ)
                @test size(∇ϕᵢ) == (ndim(f),)
                @test size(Hϕᵢ) == (ndim(f), ndim(f))
                if d == 1
                    @test all(Hϕᵢ .≈ 0)
                end
            end
        end

        @testset "First derivative matrix" begin
            Ds = derivative_matrices(f, X; derivative_idcs = [1, 2])
            @test length(Ds) == 2
            for D in Ds
                @test size(D) == (length(X), ndofs(f))
                @test nnz(D) <= length(X) * getnbasefunctions(ip)
                @test nnz(D) > 0
            end
        end

        @testset "Second derivative matrix" begin
            Ds = second_derivative_matrices(f, X; derivative_idcs = [(1, 1), (2, 2)])
            @test length(Ds) == 2
            for D in Ds
                @test size(D) == (length(X), ndofs(f))
                if d == 1
                    @test all(D .≈ 0)
                else
                    @test nnz(D) <= length(X) * getnbasefunctions(ip)
                    @test nnz(D) > 0
                end
            end
        end
    end
end
