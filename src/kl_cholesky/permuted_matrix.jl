using LinearAlgebra

export PermutedMatrix

"""
    PermutedMatrix{T, M <: AbstractMatrix{T}} <: AbstractMatrix{T}

A memory-efficient wrapper that represents a symmetrically permuted matrix without
materializing it. For a matrix `A` and permutation vector `p`, `PermutedMatrix(A, p)`
represents `A[p, p]` without forming it explicitly.

This is particularly useful for accessing elements of permuted matrices in sparse
matrix algorithms without the memory overhead of creating a full permuted copy.

# Type Parameters
- `T`: Element type of the matrix
- `M`: Type of the underlying matrix (must be <: AbstractMatrix{T})

# Fields
- `matrix::M`: The underlying matrix
- `perm::Vector{Int}`: Permutation vector (must have length equal to matrix dimensions)

# Examples
```julia
A = [1 2 3; 4 5 6; 7 8 9]
p = [3, 1, 2]
PA = PermutedMatrix(A, p)

# Access individual elements (applies permutation to both indices)
PA[1, 1] == A[3, 3]  # true
PA[1, 2] == A[3, 1]  # true

# Access blocks
PA[1:2, 1:2] == A[[3, 1], [3, 1]]  # true

# Convert to explicit matrix
to_matrix(PA) == A[p, p]  # true
```
"""
struct PermutedMatrix{T, M <: AbstractMatrix{T}} <: AbstractMatrix{T}
    matrix::M
    perm::Vector{Int}

    function PermutedMatrix(matrix::M, perm::Vector{Int}) where {T, M <: AbstractMatrix{T}}
        n, m = size(matrix)
        if n != m
            throw(ArgumentError("PermutedMatrix requires a square matrix, got size ($n, $m)"))
        end
        if length(perm) != n
            throw(
                ArgumentError(
                    "Permutation vector length ($(length(perm))) must match matrix dimension ($n)"
                ),
            )
        end
        if !isperm(perm)
            throw(ArgumentError("Permutation vector must be a valid permutation"))
        end
        return new{T, M}(matrix, perm)
    end
end

# AbstractArray interface
Base.size(pm::PermutedMatrix) = size(pm.matrix)
Base.axes(pm::PermutedMatrix) = axes(pm.matrix)
Base.IndexStyle(::Type{<:PermutedMatrix}) = IndexCartesian()

# Scalar indexing - applies permutation to both row and column
Base.@propagate_inbounds function Base.getindex(pm::PermutedMatrix, i::Int, j::Int)
    @boundscheck checkbounds(pm, i, j)
    return pm.matrix[pm.perm[i], pm.perm[j]]
end

# Vector indexing - applies permutation to both row and column indices
Base.@propagate_inbounds function Base.getindex(
        pm::PermutedMatrix, I::AbstractVector, J::AbstractVector
    )
    @boundscheck checkbounds(pm, I, J)
    return pm.matrix[pm.perm[I], pm.perm[J]]
end

# Colon indexing - all rows, specific column
Base.@propagate_inbounds function Base.getindex(pm::PermutedMatrix, ::Colon, j::Int)
    @boundscheck checkbounds(pm, :, j)
    return pm.matrix[pm.perm, pm.perm[j]]
end

Base.@propagate_inbounds function Base.getindex(
        pm::PermutedMatrix, ::Colon, J::AbstractVector
    )
    @boundscheck checkbounds(pm, :, J)
    return pm.matrix[pm.perm, pm.perm[J]]
end

# Colon indexing - specific row, all columns
Base.@propagate_inbounds function Base.getindex(pm::PermutedMatrix, i::Int, ::Colon)
    @boundscheck checkbounds(pm, i, :)
    return pm.matrix[pm.perm[i], pm.perm]
end

Base.@propagate_inbounds function Base.getindex(
        pm::PermutedMatrix, I::AbstractVector, ::Colon
    )
    @boundscheck checkbounds(pm, I, :)
    return pm.matrix[pm.perm[I], pm.perm]
end

# Both colon - return full permuted matrix
Base.@propagate_inbounds function Base.getindex(pm::PermutedMatrix, ::Colon, ::Colon)
    return pm.matrix[pm.perm, pm.perm]
end

function to_matrix(pm::PermutedMatrix)
    return pm.matrix[pm.perm, pm.perm]
end
