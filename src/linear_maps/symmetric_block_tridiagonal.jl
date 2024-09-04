using LinearAlgebra, LinearMaps
import SparseArrays: sparse

import Base: size

export SymmetricBlockTridiagonalMap

"""
    SymmetricBlockTridiagonalMap

A linear map representing a symmetric block tridiagonal matrix
with diagonal blocks `diagonal_blocks` and lower off-diagonal blocks
`off_diagonal_blocks`.
"""
struct SymmetricBlockTridiagonalMap{T} <: LinearMap{T}
    diagonal_blocks::Vector{<:LinearMap{T}}
    off_diagonal_blocks::Vector{<:LinearMap{T}}
    size::Dims{2}

    function SymmetricBlockTridiagonalMap(
        diagonal_blocks::Vector{<:LinearMap{T}},
        off_diagonal_blocks::Vector{<:LinearMap{T}},
    ) where {T}
        N = sum(map(x -> Base.size(x, 1), diagonal_blocks))
        sz = Dims((N, N))
        if length(off_diagonal_blocks) != length(diagonal_blocks) - 1
            throw(ArgumentError("size mismatch"))
        end
        # Size checks
        for (i, block) in enumerate(diagonal_blocks)
            if i > 1
                size(block, 1) == Base.size(off_diagonal_blocks[i-1], 1) ||
                    throw(ArgumentError("size mismatch"))
            end
        end
        new{T}(diagonal_blocks, off_diagonal_blocks, sz)
    end
end

size(d::SymmetricBlockTridiagonalMap) = d.size

function LinearMaps._unsafe_mul!(y, A::SymmetricBlockTridiagonalMap, x::AbstractVector)
    y .= 0
    start = 1
    stop = 0
    for (i, block) in enumerate(A.diagonal_blocks)
        stop += Base.size(block, 1)
        y[start:stop] .= block * x[start:stop]
        if i > 1
            off_block = A.off_diagonal_blocks[i-1]
            off_block_size = Base.size(off_block, 2)
            y[start:stop] .+= off_block * x[start-off_block_size:start-1]
        end
        if i < length(A.diagonal_blocks)
            off_block = A.off_diagonal_blocks[i]
            off_block_size = Base.size(off_block', 2)
            y[start:stop] .+= off_block' * x[stop+1:stop+off_block_size]
        end
        start = stop + 1
    end
    return y
end

function sparse(A::SymmetricBlockTridiagonalMap)
    M = spzeros(Base.size(A))
    start = 1
    stop = 0
    for (i, block) in enumerate(A.diagonal_blocks)
        stop += Base.size(block, 1)
        M[start:stop, start:stop] = sparse(block)
        if i > 1
            off_block = A.off_diagonal_blocks[i-1]
            off_block_size = Base.size(off_block, 2)
            M[start:stop, start-off_block_size:start-1] = sparse(off_block)
        end
        if i < length(A.diagonal_blocks)
            off_block = A.off_diagonal_blocks[i]
            off_block_size = Base.size(off_block', 2)
            M[start:stop, stop+1:stop+off_block_size] = sparse(off_block')
        end
        start = stop + 1
    end
    return M
end
