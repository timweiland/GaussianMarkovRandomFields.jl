using Kronecker: Kronecker, kronecker, AbstractKroneckerProduct, getmatrices
using LinearAlgebra
using SparseArrays

export BlockDiagonalPrecision

# Structured precision representations.
#
# Principle: `precision_matrix` returns the most structured representation it
# can, and structure survives until it is provably destroyed. The two
# destruction seams are (1) `_ensure_sparse` lowering — where values enter a
# workspace pattern or a LinearSolve factorization — and (2) the posterior
# update `Q_post = Q_prior - H`. Everything scalar (log-determinants, quadratic
# forms) is computed by dispatch on the structured type, at factor scale,
# without ever factorizing the joint matrix.
#
# `SeparableModel` returns a lazy `Kronecker.KroneckerProduct`;
# `CombinedModel` returns a `BlockDiagonalPrecision` when any component is
# itself structured (the block-diagonal wrapper is the conduit that lets a
# Kronecker block survive composition).

"""
    BlockDiagonalPrecision(blocks::AbstractMatrix...)

Lazy block-diagonal matrix over heterogeneous square blocks (sparse,
`Diagonal`, or Kronecker blocks). Used as the precision representation of a
`CombinedModel` whenever at least one component precision is structured, so
that per-block structure (e.g. a Kronecker block from a `SeparableModel`
component) survives composition.

Lower to a concrete sparse matrix with `sparse` / `_ensure_sparse`.
"""
struct BlockDiagonalPrecision{T, Bs <: Tuple{Vararg{AbstractMatrix}}} <: AbstractMatrix{T}
    blocks::Bs
    offsets::Vector{Int}  # length nblocks + 1; block i occupies offsets[i]+1:offsets[i+1]

    function BlockDiagonalPrecision(blocks::Tuple{Vararg{AbstractMatrix}})
        isempty(blocks) && throw(ArgumentError("BlockDiagonalPrecision requires at least one block"))
        for B in blocks
            size(B, 1) == size(B, 2) ||
                throw(ArgumentError("BlockDiagonalPrecision blocks must be square, got $(size(B))"))
        end
        T = promote_type(map(eltype, blocks)...)
        offsets = cumsum([0; [size(B, 1) for B in blocks]])
        return new{T, typeof(blocks)}(blocks, offsets)
    end
end

BlockDiagonalPrecision(blocks::AbstractMatrix...) = BlockDiagonalPrecision(blocks)

Base.size(B::BlockDiagonalPrecision) = (B.offsets[end], B.offsets[end])

function Base.getindex(B::BlockDiagonalPrecision{T}, i::Int, j::Int) where {T}
    @boundscheck checkbounds(B, i, j)
    bi = searchsortedlast(B.offsets, i - 1)
    bj = searchsortedlast(B.offsets, j - 1)
    bi == bj || return zero(T)
    return T(B.blocks[bi][i - B.offsets[bi], j - B.offsets[bj]])
end

# Block-wise matvec. The generic `*(::AbstractMatrix, ::AbstractVector)`
# routes through this 3-arg `mul!`, so no `*` method is defined here — a
# `*(::BlockDiagonalPrecision, ::AbstractVector)` would be ambiguous against
# the special-vector `*` methods of FillArrays/NamedDims/SciMLBase.
function LinearAlgebra.mul!(y::AbstractVector, B::BlockDiagonalPrecision, x::AbstractVector)
    length(x) == size(B, 2) || throw(DimensionMismatch("matvec size mismatch"))
    for (k, block) in enumerate(B.blocks)
        rng = (B.offsets[k] + 1):B.offsets[k + 1]
        mul!(view(y, rng), block, view(x, rng))
    end
    return y
end

# --- Structure trait ---

"""
    _is_structured(Q) -> Bool

Whether `Q` is a lazy structured precision representation (Kronecker product
or block-diagonal) that the structured prior fast path knows how to exploit.
Plain sparse/dense/`Diagonal` matrices are not "structured" in this sense —
they are already their own best representation.
"""
_is_structured(::AbstractMatrix) = false
_is_structured(::AbstractKroneckerProduct) = true
_is_structured(::BlockDiagonalPrecision) = true

# --- Factor flattening ---

"""
    _structure_factors(K) -> Tuple

Flatten a (possibly nested) Kronecker product into its ordered factor tuple.
`kronecker(A, B, C)` nests right-associatively; this recovers `(A, B, C)`.
"""
_structure_factors(K::AbstractKroneckerProduct) =
    (_structure_factors(getmatrices(K)[1])..., _structure_factors(getmatrices(K)[2])...)
_structure_factors(A::AbstractMatrix) = (A,)

# --- Lowering (the destruction seam) ---
# `_ensure_sparse`'s generic methods live in latent_models/combined.jl; the
# methods below extend the same function for the lazy structured types.
# `_blockdiag` (also from combined.jl) is only called at runtime, so the
# include order does not matter.

function _ensure_sparse(K::AbstractKroneckerProduct)
    factors = map(_ensure_sparse, _structure_factors(K))
    return foldl(kron, factors)
end

# Deterministic Diagonal lowering: every diagonal position is stored,
# regardless of values. Structured-prior caching relies on lowering being a
# pure function of input *patterns*; a value-sensitive conversion that drops
# exact zeros would silently invalidate the cached scatter map.
function _ensure_sparse(D::Diagonal)
    n = size(D, 1)
    return SparseMatrixCSC(n, n, collect(1:(n + 1)), collect(1:n), Vector(D.diag))
end

_ensure_sparse(B::BlockDiagonalPrecision) = _blockdiag(map(_ensure_sparse, B.blocks)...)

SparseArrays.sparse(B::BlockDiagonalPrecision) = _ensure_sparse(B)

# --- Cacheless structured log-determinant ---
#
# `logdet(A ⊗ B) = size(B, 1) * logdet(A) + size(A, 1) * logdet(B)`, applied
# recursively; block-diagonal log-determinants sum over blocks. Sparse leaves
# factorize — but a *factor*, not the joint matrix. The cached variant (which
# reuses per-factor `GMRFWorkspace` engines across evaluations) lives in
# structured/prior_cache.jl; this variant serves one-shot paths like
# `prior_logdensity`.

function _structured_logdet(K::AbstractKroneckerProduct)
    A, B = getmatrices(K)
    return size(B, 1) * _structured_logdet(A) + size(A, 1) * _structured_logdet(B)
end

_structured_logdet(B::BlockDiagonalPrecision) = sum(_structured_logdet, B.blocks)

# Exact and eltype-generic (works for ForwardDiff.Dual values as-is).
_structured_logdet(D::Diagonal) = sum(log, D.diag)

# Funnel: lower any other leaf (SymTridiagonal from AR1/RW1, plain sparse, …)
# to sparse and factorize. The sparse leaf is deliberately typed
# `<:AbstractFloat`: a `SparseMatrixCSC{<:ForwardDiff.Dual}` must NOT silently
# hit a value-stripping path — the ForwardDiff extension provides the exact
# Dual method (primal logdet + selected-inverse tangent).
_structured_logdet(Q::AbstractMatrix) = _sparse_leaf_logdet(_ensure_sparse(Q))

_sparse_leaf_logdet(Q::SparseMatrixCSC{<:AbstractFloat}) =
    logdet(cholesky(Symmetric(Q)))

# --- Cacheless structured quadratic form ---

"""
    _structured_quadform(Q, r) -> Real

`r' Q r` without materializing `Q`. Kronecker products use the vec-trick
matvec; block-diagonals multiply block-wise.
"""
_structured_quadform(Q::AbstractMatrix, r::AbstractVector) = dot(r, Q * r)
