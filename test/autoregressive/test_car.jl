using GaussianMarkovRandomFields
using LinearAlgebra
using SparseArrays

@testset "Conditional Autoregressive Models (CARs)" begin
    N = 300
    W = spzeros(N, N)
    for i = 1:N
        for k in [-2, -1, 1, 2]
            j = i + k
            if 1 <= j <= N
                W[i, j] = 1.0 / abs(k)
            end
        end
    end
    ϕ = 0.995
    xs = range(0, 1, length = N)
    μ = [ϕ^(i - 1) for i in eachindex(xs)]

    x = generate_car_model(W, 0.9; μ = μ, σ = 0.001)

    @test mean(x) ≈ μ
    # Test sparsity pattern of precision matrix
    Q = to_matrix(precision_map(x))
    @test nnz(Q) == nnz(W) + N

    @test_throws ArgumentError generate_car_model(W, 1.0)
    @test_throws ArgumentError generate_car_model(W, -0.1)
end
