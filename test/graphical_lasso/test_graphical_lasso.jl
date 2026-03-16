using GaussianMarkovRandomFields
using GraphicalLassoQUIC
using LinearAlgebra
using SparseArrays
using Random

function glasso_obj(Θ, S, λ)
    offdiag = sum(abs, Θ) - sum(abs, diag(Θ))
    return dot(Θ, S) - logdet(Symmetric(Θ)) + λ * offdiag
end

@testset "graphical lasso" begin
    rng = Random.MersenneTwister(12345)

    n = 500
    m = 5000
    λ = 0.001

    # random sparse precision matrix
    A = sprand(rng, n, n, 0.05)
    A = A + A'

    for i in 1:n
        A[i, i] = sum(abs, A[:, i]) + 1.0
    end

    M = Hermitian(A, :L)

    # generate m samples
    X = copy(transpose(cholesky(M).L \ randn(rng, n, m)))
    S = X' * X / m

    # our implementation
    gmrf = graphical_lasso(X, λ)
    Θ_ours = Matrix(precision_matrix(gmrf))

    # restricted graphical lasso
    Λ = copy(A); Λ.nzval .= λ
    gmrf = graphical_lasso(X, Λ)
    Θ_rest = Matrix(precision_matrix(gmrf))

    # QUIC
    Θ_quic = GraphicalLassoQUIC.QUIC(S, fill(λ, n, n))

    # compare objectives
    obj_ours = glasso_obj(Θ_ours, S, λ)
    obj_rest = glasso_obj(Θ_rest, S, λ)
    obj_quic = glasso_obj(Θ_quic, S, λ)

    @test obj_ours ≈ obj_quic rtol=1e-2
    @test obj_ours < obj_rest
end
