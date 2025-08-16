using LinearAlgebra, LinearMaps
import SparseArrays: sparse
import LinearAlgebra: issymmetric

import Base: size

export SymmetricBlockTridiagonalMap

"""
    SymmetricBlockTridiagonalMap(
        diagonal_blocks::Tuple{LinearMap{T},Vararg{LinearMap{T},ND}},
        off_diagonal_blocks::Tuple{LinearMap{T},Vararg{LinearMap{T},NOD}},
    )

A linear map representing a symmetric block tridiagonal matrix
with diagonal blocks `diagonal_blocks` and lower off-diagonal blocks
`off_diagonal_blocks`.
"""
struct SymmetricBlockTridiagonalMap{T} <: LinearMap{T}
    diagonal_blocks::Tuple{Vararg{LinearMap{T}}}
    off_diagonal_blocks::Tuple{Vararg{LinearMap{T}}}
    size::Dims{2}

    function SymmetricBlockTridiagonalMap(
            diagonal_blocks::Tuple{LinearMap{T}, Vararg{LinearMap{T}, ND}},
            off_diagonal_blocks::Tuple{LinearMap{T}, Vararg{LinearMap{T}, NOD}},
        ) where {T, ND, NOD}
        N = sum(map(x -> Base.size(x, 1), diagonal_blocks))
        sz = Dims((N, N))
        if NOD != ND - 1
            throw(ArgumentError("size mismatch"))
        end
        # Size checks
        for (i, block) in enumerate(diagonal_blocks)
            if i > 1
                size(block, 1) == Base.size(off_diagonal_blocks[i - 1], 1) ||
                    throw(ArgumentError("size mismatch"))
            end
        end
        return new{T}(diagonal_blocks, off_diagonal_blocks, sz)
    end

    function SymmetricBlockTridiagonalMap(
            diagonal_blocks::Tuple{LinearMap{T}},
            off_diagonal_blocks::Tuple{},
        ) where {T}
        sz = Dims(Base.size(diagonal_blocks[1]))
        return new{T}(diagonal_blocks, off_diagonal_blocks, sz)
    end
end

size(d::SymmetricBlockTridiagonalMap) = d.size
LinearAlgebra.issymmetric(::SymmetricBlockTridiagonalMap) = true

function LinearMaps._unsafe_mul!(y, A::SymmetricBlockTridiagonalMap, x::AbstractVector)
    y .= 0
    start = 1
    stop = 0
    for (i, block) in enumerate(A.diagonal_blocks)
        stop += Base.size(block, 1)
        y[start:stop] .= block * x[start:stop]
        if i > 1
            off_block = A.off_diagonal_blocks[i - 1]
            off_block_size = Base.size(off_block, 2)
            y[start:stop] .+= off_block * x[(start - off_block_size):(start - 1)]
        end
        if i < length(A.diagonal_blocks)
            off_block = A.off_diagonal_blocks[i]
            off_block_size = Base.size(off_block', 2)
            y[start:stop] .+= off_block' * x[(stop + 1):(stop + off_block_size)]
        end
        start = stop + 1
    end
    return y
end

function sparse(A::SymmetricBlockTridiagonalMap)
    M = spzeros(Base.size(A))
    Is = []
    Js = []
    Vs = []
    start = 1
    stop = 0
    for (i, block) in enumerate(A.diagonal_blocks)
        stop += Base.size(block, 1)
        diag_I, diag_J, diag_V = findnz(sparse(to_matrix(block)))
        diag_I .+= start - 1
        diag_J .+= start - 1
        push!(Is, diag_I)
        push!(Js, diag_J)
        push!(Vs, diag_V)
        if i > 1
            off_block = to_matrix(A.off_diagonal_blocks[i - 1])
            off_block_size = Base.size(off_block, 2)
            off_diag_I, off_diag_J, off_diag_V = findnz(sparse(off_block))
            off_diag_I .+= start - 1
            off_diag_J .+= start - off_block_size - 1
            push!(Is, off_diag_I)
            push!(Js, off_diag_J)
            push!(Vs, off_diag_V)
        end
        start = stop + 1
    end
    M = sparse(vcat(Is...), vcat(Js...), vcat(Vs...), Base.size(A, 1), Base.size(A, 2))
    return Symmetric(M, :L)
end
