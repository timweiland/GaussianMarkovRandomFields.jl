using CliqueTrees: Multifrontal
using CliqueTrees.Multifrontal: ChordalCholesky, complete!
using LinearAlgebra: Hermitian, HermOrSym, mul!
using Statistics: mean
using SparseArrays: SparseMatrixCSC, sparse, nzrange, rowvals, nonzeros

export graphical_lasso

"""
    graphical_lasso(X::AbstractMatrix, threshold; blocksize::Int=256, shift=zero(T), alg=nothing)

Learn a GMRF from data by solving the graphical lasso problem:

    maximize  log det Ω - tr(ΣΩ) - λ ‖Ω‖₁
    such that  Ω is positive definite

where Σ is the sample covariance and Ω is the GMRF's precision matrix.

`threshold` can be a scalar `λ` for uniform penalization, or a `SparseMatrixCSC`
for per-entry penalties within a given sparsity pattern (restricted graphical lasso).

`shift` adds a constant to the diagonal of the sample covariance as regularization
to improve convergence.
"""
function graphical_lasso(X::AbstractMatrix{T}, threshold; blocksize::Int = 256, shift::T = zero(T), alg = nothing) where {T}
    # We will solve the graphical lasso problem using the approach in
    # Zhang, Fattahi, and Sojoudi: "Large-Scale Sparse Inverse Covariance
    # Estimation via Thresholding and Max-Det Matrix Completion".
    #
    # First, we construct a truncated covariance matrix from the sample covariance matrix C.
    # This is derived from the sample covariance matrix Σ via the following formula:
    #
    #   Cij = { Σij - λ if Σij > λ
    #         { Σij + λ if Σij < -λ
    #         { 0       else
    #
    C, μ = soft_threshold_cov(X, threshold; blocksize, shift)
    #
    # Next, we solve the maximum-determinant positive-definite completion problem:
    #
    #   maximize  log det Σ
    #   such that Σ is positive definite and
    #             Σij = Cij for for each structural nonzero (i,j) of C
    #
    return graphical_lasso(C, μ; alg)
end

function graphical_lasso(C::HermOrSym{T}, μ::AbstractVector{T}; alg = nothing) where {T}
    F = complete!(ChordalCholesky(C), C)
    P = sparse(F)
    return GMRF(μ, P, alg)
end

function soft_threshold_cov(X::AbstractMatrix{T}, threshold::T; blocksize::Int = 256, shift::T = zero(T)) where {T}
    m, n = size(X)

    μ = vec(mean(X, dims = 1))

    colptr = Vector{Int}(undef, n + 1)
    rowval = Int[]
    nzval = T[]

    block = Matrix{T}(undef, n, blocksize)

    for jstart in 1:blocksize:n
        jstop = min(jstart + blocksize - 1, n)
        jsize = jstop - jstart + 1
        jblock = view(block, :, 1:jsize)
        mul!(jblock, X', view(X, :, jstart:jstop), 1 / m, false)

        for (k, j) in enumerate(jstart:jstop)
            colptr[j] = length(rowval) + 1
            μj = μ[j]
            Cjj = jblock[j, k] - μj * μj + shift

            push!(rowval, j)
            push!(nzval, Cjj)

            for i in (j + 1):n
                Cij = jblock[i, k] - μ[i] * μj

                if Cij > threshold
                    push!(rowval, i)
                    push!(nzval, Cij - threshold)
                elseif Cij < -threshold
                    push!(rowval, i)
                    push!(nzval, Cij + threshold)
                end
            end
        end
    end

    colptr[n + 1] = length(rowval) + 1
    Y = SparseMatrixCSC{T, Int}(n, n, colptr, rowval, nzval)
    return Hermitian(Y, :L), μ
end

function soft_threshold_cov(X::AbstractMatrix{T}, threshold::SparseMatrixCSC{T}; blocksize::Int = 256, shift::T = zero(T)) where {T}
    m, n = size(X)

    μ = vec(mean(X, dims = 1))

    colptr = Vector{Int}(undef, n + 1)
    rowval = Int[]
    nzval = T[]

    block = Matrix{T}(undef, n, blocksize)

    for jstart in 1:blocksize:n
        jstop = min(jstart + blocksize - 1, n)
        jsize = jstop - jstart + 1
        jblock = view(block, :, 1:jsize)
        mul!(jblock, X', view(X, :, jstart:jstop), 1 / m, false)

        for (k, j) in enumerate(jstart:jstop)
            colptr[j] = length(rowval) + 1
            μj = μ[j]
            Cjj = jblock[j, k] - μj * μj + shift

            push!(rowval, j)
            push!(nzval, Cjj)

            for p in nzrange(threshold, j)
                i = rowvals(threshold)[p]

                if i > j
                    Tij = nonzeros(threshold)[p]
                    Cij = jblock[i, k] - μ[i] * μj

                    if Cij > Tij
                        push!(rowval, i)
                        push!(nzval, Cij - Tij)
                    elseif Cij < -Tij
                        push!(rowval, i)
                        push!(nzval, Cij + Tij)
                    end
                end
            end
        end
    end

    colptr[n + 1] = length(rowval) + 1
    return Hermitian(SparseMatrixCSC{T, Int}(n, n, colptr, rowval, nzval), :L), μ
end
