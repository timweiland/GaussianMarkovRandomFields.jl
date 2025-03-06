import LinearAlgebra: ldiv!
import Base: \, size, show

export TridiagonalBlockGaussSeidelPreconditioner,
    TridiagSymmetricBlockGaussSeidelPreconditioner

@doc raw"""
    TridiagonalBlockGaussSeidelPreconditioner{T}(D_blocks, L_blocks)
    TridiagonalBlockGaussSeidelPreconditioner{T}(D⁻¹_blocks, L_blocks)

Block Gauss-Seidel preconditioner for block tridiagonal matrices.
For a matrix given by

```math
A = \begin{bmatrix}
D₁ & L₁ᵀ & 0 & \cdots & 0 \\
L₁ & D₂ & L₂ᵀ & 0 & \cdots \\
0 & L₂ & D₃ & L₃ᵀ & \cdots \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & \cdots & 0 & Lₙ₋₁ & Lₙ
\end{bmatrix}
```

this preconditioner is given by

```math
P = \begin{bmatrix}
D₁ & 0 & \cdots & 0 \\
L₁ & D₂ & 0 & \cdots \\
0 & L₂ & D₃ & \cdots \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & \cdots & 0 & Lₙ₋₁ & Dₙ
\end{bmatrix}
```

Solving linear systems with the preconditioner is made efficient through block
forward / backward substitution.
The diagonal blocks must be inverted. As such, they may be specified
 1. directly as matrices: in this case they will be transformed into
    `FullCholeskyPreconditioner`s.
 2. in terms of their invertible preconditioners
"""
struct TridiagonalBlockGaussSeidelPreconditioner{T} <: AbstractPreconditioner{T}
    D⁻¹_blocks::Tuple{Vararg{AbstractPreconditioner{T}}}
    L_blocks::Tuple{Vararg{AbstractMatrix{T}}}

    function TridiagonalBlockGaussSeidelPreconditioner(
        D_blocks::Tuple{AbstractMatrix{T},Vararg{AbstractMatrix{T},ND}},
        L_blocks::Tuple{AbstractMatrix{T},Vararg{AbstractMatrix{T},NOD}},
    ) where {T,ND,NOD}
        NOD == ND - 1 || throw(ArgumentError("size mismatch"))
        D⁻¹_blocks = FullCholeskyPreconditioner.(D_blocks)
        new{T}(D⁻¹_blocks, L_blocks)
    end

    function TridiagonalBlockGaussSeidelPreconditioner(
        D_blocks::Tuple{AbstractMatrix{T}},
        L_blocks::Tuple{},
    ) where {T}
        new{T}(FullCholeskyPreconditioner.(D_blocks), L_blocks)
    end

    function TridiagonalBlockGaussSeidelPreconditioner(
        D⁻¹_blocks::Tuple{AbstractPreconditioner{T},Vararg{AbstractPreconditioner{T},ND}},
        L_blocks::Tuple{AbstractMatrix{T},Vararg{AbstractMatrix{T},NOD}},
    ) where {T,ND,NOD}
        NOD == ND - 1 || throw(ArgumentError("size mismatch"))
        new{T}(D⁻¹_blocks, L_blocks)
    end

    function TridiagonalBlockGaussSeidelPreconditioner(
        D⁻¹_blocks::Tuple{AbstractPreconditioner{T}},
        L_blocks::Tuple{},
    ) where {T}
        new{T}(D⁻¹_blocks, L_blocks)
    end
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


"""
    TridiagSymmetricBlockGaussSeidelPreconditioner{T}(D_blocks, L_blocks)
    TridiagSymmetricBlockGaussSeidelPreconditioner{T}(D⁻¹_blocks, L_blocks)

Symmetric Block Gauss-Seidel preconditioner for symmetric block tridiagonal
matrices.
For a symmetric matrix given by the block decomposition
A = L + D + Lᵀ,
where L is strictly lower triangular and D is diagonal,
this preconditioner is given by
P = (L + D) D⁻¹ (L + D)ᵀ ≈ A.

Solving linear systems with the preconditioner is made efficient through block
forward / backward substitution.
The diagonal blocks must be inverted. As such, they may be specified
 1. directly as matrices: in this case they will be transformed into
    `FullCholeskyPreconditioner`s.
 2. in terms of their invertible preconditioners
"""
struct TridiagSymmetricBlockGaussSeidelPreconditioner{T} <: AbstractPreconditioner{T}
    D⁻¹_blocks::Tuple{Vararg{AbstractPreconditioner{T}}}
    L_blocks::Tuple{Vararg{AbstractMatrix{T}}}

    function TridiagSymmetricBlockGaussSeidelPreconditioner(
        D_blocks::Tuple{AbstractMatrix{T},Vararg{AbstractMatrix{T},ND}},
        L_blocks::Tuple{AbstractMatrix{T},Vararg{AbstractMatrix{T},NOD}},
    ) where {T,ND,NOD}
        NOD == ND - 1 || throw(ArgumentError("size mismatch"))
        D⁻¹_blocks = FullCholeskyPreconditioner.(D_blocks)
        new{T}(D⁻¹_blocks, L_blocks)
    end

    function TridiagSymmetricBlockGaussSeidelPreconditioner(
        D_blocks::Tuple{AbstractMatrix{T}},
        L_blocks::Tuple{},
    ) where {T}
        new{T}(FullCholeskyPreconditioner.(D_blocks), L_blocks)
    end

    function TridiagSymmetricBlockGaussSeidelPreconditioner(
        D⁻¹_blocks::Tuple{AbstractPreconditioner{T},Vararg{AbstractPreconditioner{T},ND}},
        L_blocks::Tuple{AbstractMatrix{T},Vararg{AbstractMatrix{T},NOD}},
    ) where {T,ND,NOD}
        NOD == ND - 1 || throw(ArgumentError("size mismatch"))
        new{T}(D⁻¹_blocks, L_blocks)
    end

    function TridiagSymmetricBlockGaussSeidelPreconditioner(
        D⁻¹_blocks::Tuple{AbstractPreconditioner{T}},
        L_blocks::Tuple{},
    ) where {T}
        new{T}(D⁻¹_blocks, L_blocks)
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

# COV_EXCL_START
function show(io::IO, P::TridiagonalBlockGaussSeidelPreconditioner)
    print(
        io,
        "TridiagonalBlockGaussSeidelPreconditioner with $(length(P.D⁻¹_blocks)) blocks",
    )
end

function show(io::IO, P::TridiagSymmetricBlockGaussSeidelPreconditioner)
    print(
        io,
        "TridiagSymmetricBlockGaussSeidelPreconditioner with $(length(P.D⁻¹_blocks)) blocks",
    )
end
# COV_EXCL_STOP
