import LinearAlgebra: ldiv!
import Base: \, size, show

export TridiagonalBlockGaussSeidelPreconditioner,
    TridiagSymmetricBlockGaussSeidelPreconditioner

struct TridiagonalBlockGaussSeidelPreconditioner <: AbstractPreconditioner
    L_blocks::Vector{<:AbstractMatrix}
    D⁻¹_blocks::Vector{<:AbstractPreconditioner}

    function TridiagonalBlockGaussSeidelPreconditioner(
        L_blocks::Vector{<:AbstractMatrix},
        D_blocks::Vector{<:AbstractMatrix},
    )
        length(L_blocks) == length(D_blocks) - 1 || throw(ArgumentError("size mismatch"))
        D⁻¹_blocks = FullCholeskyPreconditioner.(D_blocks)
        new(L_blocks, D⁻¹_blocks)
    end

    function TridiagonalBlockGaussSeidelPreconditioner(
        L_blocks::Vector{<:AbstractMatrix},
        D⁻¹_blocks::Vector{<:AbstractPreconditioner},
    )
        length(L_blocks) == length(D⁻¹_blocks) - 1 || throw(ArgumentError("size mismatch"))
        new(L_blocks, D⁻¹_blocks)
    end
end

function show(io::IO, P::TridiagonalBlockGaussSeidelPreconditioner)
    print(
        io,
        "TridiagonalBlockGaussSeidelPreconditioner with $(length(P.D⁻¹_blocks)) blocks",
    )
end

function ldiv!(y, P::TridiagonalBlockGaussSeidelPreconditioner, x::AbstractVector)
    L_blocks, D⁻¹_blocks = P.L_blocks, P.D⁻¹_blocks
    start = 1
    stop = size(D⁻¹_blocks[1], 1)
    y[start:stop] .= D⁻¹_blocks[1] \ x[start:stop]
    for (L, D⁻¹) in zip(L_blocks, D⁻¹_blocks[2:end])
        prev_start = start
        start = stop + 1
        stop = start + size(D⁻¹, 1) - 1
        y[start:stop] .= D⁻¹ \ (x[start:stop] - L * y[prev_start:start-1])
    end
    return y
end

ldiv!(P::TridiagonalBlockGaussSeidelPreconditioner, x::AbstractVector) = ldiv!(x, P, x)

function \(P::TridiagonalBlockGaussSeidelPreconditioner, x::AbstractVector)
    y = similar(x)
    return ldiv!(y, P, x)
end

Base.size(P::TridiagonalBlockGaussSeidelPreconditioner) = reduce(.+, size.(P.D⁻¹_blocks))


struct TridiagSymmetricBlockGaussSeidelPreconditioner <: AbstractPreconditioner
    L_blocks::Vector{<:AbstractMatrix}
    D⁻¹_blocks::Vector{<:AbstractPreconditioner}

    function TridiagSymmetricBlockGaussSeidelPreconditioner(
        L_blocks::Vector{<:AbstractMatrix},
        D_blocks::Vector{<:AbstractMatrix},
    )
        length(L_blocks) == (length(D_blocks) - 1) || throw(ArgumentError("size mismatch"))
        D⁻¹_blocks = FullCholeskyPreconditioner.(D_blocks)
        new(L_blocks, D⁻¹_blocks)
    end

    function TridiagSymmetricBlockGaussSeidelPreconditioner(
        L_blocks::Vector{<:AbstractMatrix},
        D⁻¹_blocks::Vector{<:AbstractPreconditioner},
    )
        length(L_blocks) == (length(D⁻¹_blocks) - 1) ||
            throw(ArgumentError("size mismatch"))
        new(L_blocks, D⁻¹_blocks)
    end
end

function ldiv!(y, P::TridiagSymmetricBlockGaussSeidelPreconditioner, x::AbstractVector)
    L_blocks, D⁻¹_blocks = P.L_blocks, P.D⁻¹_blocks

    # Forward solve
    start = 1
    stop = Base.size(D⁻¹_blocks[1], 1)
    y[start:stop] .= D⁻¹_blocks[1] \ x[start:stop]
    for (L, D⁻¹) in zip(L_blocks, D⁻¹_blocks[2:end])
        prev_start = start
        start = stop + 1
        stop = start + Base.size(D⁻¹, 1) - 1
        y[start:stop] .= D⁻¹ \ (x[start:stop] - L * y[prev_start:start-1])
    end

    # Backward solve
    for (L, D⁻¹) in zip(reverse(L_blocks), reverse(D⁻¹_blocks[1:end-1]))
        prev_stop = stop
        stop = start - 1
        start = stop - Base.size(D⁻¹, 1) + 1
        y[start:stop] .= y[start:stop] - D⁻¹ \ (L' * y[stop+1:prev_stop])
    end

    return y
end

ldiv!(P::TridiagSymmetricBlockGaussSeidelPreconditioner, x::AbstractVector) = ldiv!(x, P, x)

function \(P::TridiagSymmetricBlockGaussSeidelPreconditioner, x::AbstractVector)
    y = similar(x)
    return ldiv!(y, P, x)
end

Base.size(P::TridiagSymmetricBlockGaussSeidelPreconditioner) =
    reduce(.+, size.(P.D⁻¹_blocks))
