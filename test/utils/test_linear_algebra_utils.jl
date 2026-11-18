using GaussianMarkovRandomFields, LinearAlgebra, LinearMaps, SparseArrays

@testset "linmap_sqrt, to_matrix and symmetrize" begin
    A = [4.0 1.0 0.0; 1.0 3.0 1.0; 0.0 1.0 2.0]
    B = [2.0 0.5; 0.5 1.0]
    gram(S) = Matrix(S) * Matrix(S)'

    @testset "linmap_sqrt" begin
        # A map that is not a wrapped matrix goes through the generic fallback
        A_map = LinearMap(x -> A * x, 3; issymmetric = true, isposdef = true)
        @test gram(linmap_sqrt(A_map)) ≈ A
        @test_throws "No square root available" linmap_sqrt(LinearMap(x -> A * x, 3))

        # A sum of maps has the horizontally stacked square roots as a square root
        @test gram(linmap_sqrt(LinearMap(A) + LinearMap(2.0 * A))) ≈ 3A
        @test gram(linmap_sqrt(kron(LinearMap(A), LinearMap(B)))) ≈ kron(A, B)
    end

    @testset "to_matrix" begin
        @test Matrix(to_matrix(LinearMap(2.5I, 3))) ≈ 2.5 * I(3)
        @test Matrix(to_matrix(kron(LinearMap(A), LinearMap(B)))) ≈ kron(A, B)
    end

    @testset "symmetrize" begin
        S = SymTridiagonal([2.0, 2.0], [1.0])
        @test GaussianMarkovRandomFields.symmetrize(S) === S
        T = Tridiagonal([1.0], [2.0, 2.0], [1.0])
        @test GaussianMarkovRandomFields.symmetrize(T) == SymTridiagonal([2.0, 2.0], [1.0])
    end
end
