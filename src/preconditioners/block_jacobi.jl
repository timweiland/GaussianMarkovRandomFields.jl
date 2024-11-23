import LinearAlgebra: ldiv!
import Base: \, size

export BlockJacobiPreconditioner

"""
    BlockJacobiPreconditioner

A preconditioner that uses a block Jacobi preconditioner, i.e. P = diag(A₁, A₂, ...),
where each Aᵢ is a preconditioner for a block of the matrix.
"""
struct BlockJacobiPreconditioner{T} <: AbstractPreconditioner{T}
    blocks::Vector{<:AbstractPreconditioner{T}}
end

function ldiv!(y, P::BlockJacobiPreconditioner, x::AbstractVector)
    start = 1
    for block in P.blocks
        stop = start + size(block.cho, 1) - 1
        y[start:stop] .= ldiv!(y[start:stop], block, x[start:stop])
        start = stop + 1
    end
    return y
end

ldiv!(P::BlockJacobiPreconditioner, x::AbstractVector) = ldiv!(x, P, x)

function \(P::BlockJacobiPreconditioner, x::AbstractVector)
    y = similar(x)
    return ldiv!(y, P, x)
end

Base.size(P::BlockJacobiPreconditioner) = reduce(.+, size.(P.blocks))
Base.size(P::BlockJacobiPreconditioner, i) = sum(size(b, i) for b in P.blocks)
