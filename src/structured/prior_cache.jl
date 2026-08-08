using LinearAlgebra
using SparseArrays

# Per-workspace cache for structured (Kronecker / block-diagonal) priors.
#
# The joint `GMRFWorkspace` is the posterior's home: its single factor slot
# should only ever hold `Q_post`. The prior's factorization-shaped questions
# (log-determinant, selected-inverse diagonals, Dual tangents) are answered at
# *factor* scale by small per-factor `GMRFWorkspace` engines, cached here so
# their symbolic analyses are reused across hyperparameter evaluations —
# the same engine/view split as the joint workspace, applied recursively.

# --- Op-cache tree (mirrors the structured precision's shape) ---

abstract type PrecisionOpCache end

"Closed-form leaf: a `Diagonal` precision block. No factorization state."
struct DiagonalOpCache <: PrecisionOpCache
    n::Int
end

"General sparse leaf: a small factor engine with reusable symbolic analysis."
struct SparseOpCache <: PrecisionOpCache
    engine::GMRFWorkspace
end

"Kronecker node. `dims[i]` is the row dimension of factor `i`."
struct KroneckerOpCache <: PrecisionOpCache
    factors::Vector{PrecisionOpCache}
    dims::Vector{Int}
end

"Block-diagonal node."
struct BlockDiagOpCache <: PrecisionOpCache
    blocks::Vector{PrecisionOpCache}
end

# Primal-value extraction for engine construction. The ForwardDiff extension
# adds a method for `Dual` so that a first evaluation with Dual-valued
# hyperparameters can still seed Float64 factor engines.
_primal_float(x::Real) = Float64(x)

function _primal_sparse(Q::SparseMatrixCSC)
    return SparseMatrixCSC(Q.m, Q.n, Q.colptr, Q.rowval, map(_primal_float, Q.nzval))
end

_build_op_cache(D::Diagonal) = DiagonalOpCache(size(D, 1))

function _build_op_cache(K::AbstractKroneckerProduct)
    factors = _structure_factors(K)
    return KroneckerOpCache(
        PrecisionOpCache[_build_op_cache(f) for f in factors],
        Int[size(f, 1) for f in factors],
    )
end

function _build_op_cache(B::BlockDiagonalPrecision)
    return BlockDiagOpCache(PrecisionOpCache[_build_op_cache(b) for b in B.blocks])
end

_build_op_cache(Q::AbstractMatrix) =
    SparseOpCache(GMRFWorkspace(_primal_sparse(_ensure_sparse(Q))))

# --- Structural validation (per evaluation, before any cached use) ---
#
# Hyperparameter-dependent sparsity patterns exist (sparse scalar ops can drop
# stored zeros), so a cache is only reused when the structured precision's
# shape AND every leaf pattern match. Lowering (kron / blockdiag / Diagonal→
# sparse) is a pure function of the input patterns, so matching leaves imply
# the lowered joint matches the scatter map — no joint-sized check needed.

_op_cache_matches(D::Diagonal, c::DiagonalOpCache) = size(D, 1) == c.n
_op_cache_matches(::AbstractMatrix, ::PrecisionOpCache) = false

function _op_cache_matches(K::AbstractKroneckerProduct, c::KroneckerOpCache)
    factors = _structure_factors(K)
    length(factors) == length(c.factors) || return false
    return all(_op_cache_matches(f, cf) for (f, cf) in zip(factors, c.factors))
end

function _op_cache_matches(B::BlockDiagonalPrecision, c::BlockDiagOpCache)
    length(B.blocks) == length(c.blocks) || return false
    return all(_op_cache_matches(b, cb) for (b, cb) in zip(B.blocks, c.blocks))
end

_op_cache_matches(Q::AbstractMatrix, c::SparseOpCache) =
    _same_pattern(_ensure_sparse(Q), c.engine.Q)

# --- The per-workspace cache ---

"""
    StructuredPriorCache

Cached state for evaluating a structured prior against a fixed joint
workspace pattern: the op-cache tree of per-factor engines plus the scatter
map from the lowered joint precision's nonzero positions into the workspace
pattern (`snapshot.nzval[scatter[k]] = J.nzval[k]`).

Self-contained: a `StructuredPriorGMRF` holds a direct reference, so a cache
remains valid for the priors built from it even after the workspace's
registry slot (`ws.prior_cache`) has been rebuilt for a different model.
"""
struct StructuredPriorCache
    op::PrecisionOpCache
    scatter::Vector{Int}
end

function StructuredPriorCache(Q::AbstractMatrix, ws::GMRFWorkspace)
    op = _build_op_cache(Q)
    J = _ensure_sparse(Q)
    return StructuredPriorCache(op, _build_scatter(J, ws.Q))
end

"""
    _build_scatter(J::SparseMatrixCSC, W::SparseMatrixCSC) -> Vector{Int}

Index map from `J`'s stored positions to `W`'s stored positions, in `J`'s CSC
order. Throws if `J` has a stored entry outside `W`'s pattern (which would
silently lose prior mass). `W`'s pattern may be any superset of `J`'s — extra
positions (e.g. observation-Hessian padding) are simply never written.
"""
function _build_scatter(J::SparseMatrixCSC, W::SparseMatrixCSC)
    size(J) == size(W) ||
        throw(DimensionMismatch("prior precision has size $(size(J)) but workspace expects $(size(W))."))
    scatter = Vector{Int}(undef, nnz(J))
    J_rows = rowvals(J)
    W_rows = rowvals(W)
    k = 0
    @inbounds for col in 1:size(J, 2)
        w_ptr = first(nzrange(W, col))
        w_end = last(nzrange(W, col))
        for j_idx in nzrange(J, col)
            j_row = J_rows[j_idx]
            while w_ptr <= w_end && W_rows[w_ptr] < j_row
                w_ptr += 1
            end
            if w_ptr > w_end || W_rows[w_ptr] != j_row
                throw(
                    ArgumentError(
                        "Prior precision has a stored entry at ($j_row, $col) outside the workspace pattern."
                    )
                )
            end
            k += 1
            scatter[k] = w_ptr
        end
    end
    return scatter
end

"""
    _get_or_build_prior_cache!(ws::GMRFWorkspace, Q::AbstractMatrix) -> StructuredPriorCache

Return the workspace's registered `StructuredPriorCache` if it matches `Q`'s
structure and leaf patterns; otherwise build a fresh one and register it.
"""
function _get_or_build_prior_cache!(ws::GMRFWorkspace, Q::AbstractMatrix)
    pc = ws.prior_cache
    if pc isa StructuredPriorCache && _op_cache_matches(Q, pc.op)
        return pc
    end
    fresh = StructuredPriorCache(Q, ws)
    ws.prior_cache = fresh
    return fresh
end

"""
    _structured_snapshot(Q, cache, ws) -> SparseMatrixCSC

Materialize the structured precision's values onto the workspace pattern
(shared `colptr`/`rowval`, fresh `nzval` — the same snapshot convention as
`_snapshot_Q`). Positions of the workspace pattern not covered by the prior
remain structurally zero, which is exactly the semantics of
`_pad_to_workspace_pattern`.
"""
function _structured_snapshot(Q::AbstractMatrix, cache::StructuredPriorCache, ws::GMRFWorkspace)
    J = _ensure_sparse(Q)
    length(cache.scatter) == nnz(J) ||
        throw(ArgumentError("structured prior cache is stale: lowered pattern size changed."))
    T = eltype(J)
    nz = zeros(T, length(ws.Q.nzval))
    J_nz = nonzeros(J)
    scatter = cache.scatter
    @inbounds for k in eachindex(scatter)
        nz[scatter[k]] = J_nz[k]
    end
    return SparseMatrixCSC(ws.Q.m, ws.Q.n, ws.Q.colptr, ws.Q.rowval, nz)
end

# --- Cached structured log-determinant ---

function _structured_logdet(K::AbstractKroneckerProduct, c::KroneckerOpCache)
    factors = _structure_factors(K)
    length(factors) == length(c.factors) ||
        throw(ArgumentError("structured prior cache does not match the Kronecker structure."))
    N = prod(c.dims)
    acc = sum(
        (N ÷ c.dims[i]) * _structured_logdet(factors[i], c.factors[i])
            for i in eachindex(c.dims)
    )
    return acc
end

function _structured_logdet(B::BlockDiagonalPrecision, c::BlockDiagOpCache)
    return sum(_structured_logdet(b, cb) for (b, cb) in zip(B.blocks, c.blocks))
end

_structured_logdet(D::Diagonal, ::DiagonalOpCache) = sum(log, D.diag)

_structured_logdet(Q::AbstractMatrix, c::SparseOpCache) =
    _sparse_leaf_logdet(c.engine, _ensure_sparse(Q))

"""
    _leaf_sync!(engine::GMRFWorkspace, nzval::AbstractVector{Float64})

Load `nzval` into the factor engine unless it already holds exactly these
values (in which case any existing factorization/logdet/selinv caches remain
valid — e.g. a gradient pass at the same hyperparameters as the preceding
objective evaluation pays nothing).
"""
function _leaf_sync!(engine::GMRFWorkspace, nzval::AbstractVector{Float64})
    engine.numeric_valid && engine.Q.nzval == nzval && return nothing
    update_precision_values!(engine, nzval)
    return nothing
end

function _sparse_leaf_logdet(engine::GMRFWorkspace, Q::SparseMatrixCSC{Float64})
    _same_pattern(Q, engine.Q) ||
        throw(ArgumentError("factor pattern changed since the structured prior cache was built."))
    _leaf_sync!(engine, Q.nzval)
    return logdet(engine)
end

# --- Cached structured selected-inverse diagonal (marginal variances) ---
#
# diag((A ⊗ B)⁻¹) = diag(A⁻¹) ⊗ diag(B⁻¹), so Kronecker nodes reduce to a
# kron of per-factor selected-inverse diagonals.

function _structured_selinv_diag(K::AbstractKroneckerProduct, c::KroneckerOpCache)
    factors = _structure_factors(K)
    diags = [_structured_selinv_diag(factors[i], c.factors[i]) for i in eachindex(c.dims)]
    return foldl(kron, diags)
end

function _structured_selinv_diag(B::BlockDiagonalPrecision, c::BlockDiagOpCache)
    return reduce(vcat, [_structured_selinv_diag(b, cb) for (b, cb) in zip(B.blocks, c.blocks)])
end

_structured_selinv_diag(D::Diagonal, ::DiagonalOpCache) = 1.0 ./ D.diag

function _structured_selinv_diag(Q::AbstractMatrix, c::SparseOpCache)
    Qs = _ensure_sparse(Q)
    _same_pattern(Qs, c.engine.Q) ||
        throw(ArgumentError("factor pattern changed since the structured prior cache was built."))
    _leaf_sync!(c.engine, Qs.nzval)
    return selinv_diag(c.engine)
end

# --- Structured sampling transform ---
#
# Sampling x ~ N(0, Q⁻¹) needs any M with M Mᵀ = Q⁻¹. Each factor engine
# provides M_j via `backward_solve` (whatever fill-reducing permutation the
# backend uses, M_j M_jᵀ = Q_j⁻¹ holds), and
# (⊗ⱼ M_j)(⊗ⱼ M_j)ᵀ = ⊗ⱼ Q_j⁻¹ = Q⁻¹ — so a Kronecker sample is k rounds of
# factor-level triangular solves via the classic mode-rotation vec-trick:
# apply the currently-fastest factor columnwise, then rotate that mode to
# slowest; after k rounds the layout is restored.

"""
    _structured_sample_transform(z, Q, cache) -> Vector

Apply `M` with `M Mᵀ = Q⁻¹` to `z`, using the cached factor engines.
`z + mean` is then a sample from the structured prior.
"""
function _structured_sample_transform(
        z::AbstractVector, K::AbstractKroneckerProduct, c::KroneckerOpCache
    )
    factors = _structure_factors(K)
    x = z
    for j in length(factors):-1:1
        X = _leaf_sample_apply(c.factors[j], factors[j], reshape(x, c.dims[j], :))
        x = vec(transpose(X))
    end
    return x
end

function _structured_sample_transform(
        z::AbstractVector, B::BlockDiagonalPrecision, c::BlockDiagOpCache
    )
    x = similar(z, Float64)
    for (b, block) in enumerate(B.blocks)
        rng = (B.offsets[b] + 1):B.offsets[b + 1]
        x[rng] = _structured_sample_transform(z[rng], block, c.blocks[b])
    end
    return x
end

_structured_sample_transform(z::AbstractVector, D::Diagonal, ::DiagonalOpCache) =
    z ./ sqrt.(D.diag)

function _structured_sample_transform(z::AbstractVector, Q::AbstractMatrix, c::SparseOpCache)
    return vec(_leaf_sample_apply(c, Q, reshape(z, length(z), 1)))
end

function _leaf_sample_apply(c::SparseOpCache, Q::AbstractMatrix, Z::AbstractMatrix)
    Qs = _ensure_sparse(Q)
    _same_pattern(Qs, c.engine.Q) ||
        throw(ArgumentError("factor pattern changed since the structured prior cache was built."))
    _leaf_sync!(c.engine, Qs.nzval)
    X = Matrix{Float64}(undef, size(Z))
    for col in axes(Z, 2)
        X[:, col] = backward_solve(c.engine, Z[:, col])
    end
    return X
end

_leaf_sample_apply(::DiagonalOpCache, D::Diagonal, Z::AbstractMatrix) = Z ./ sqrt.(D.diag)
