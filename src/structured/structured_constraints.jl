using LinearAlgebra
using SparseArrays

# Structured linear-equality constraints for structured priors.
#
# When exactly one component of a separable (Kronecker) model is constrained,
# the full constraint matrix is `A = I_before ⊗ A_i ⊗ I_after`, and every
# Rue–Held correction quantity factorizes over the Kronecker structure:
#
#   A Q⁻¹ Aᵀ = (⊗_{j<i} Q_j⁻¹) ⊗ (A_i Q_i⁻¹ A_iᵀ) ⊗ (⊗_{j>i} Q_j⁻¹)
#
# so `logdet(A Q⁻¹ Aᵀ)` needs one tiny gram matrix on the constrained factor
# (m_i factor solves) plus the factor log-determinants that the structured
# prior already caches — no joint factorization, no m joint solves, and no
# dense m×N QR for redundancy removal (a single component's constraint block
# is full row rank by construction, and its Kronecker expansion with
# identities preserves that).

"""
    KroneckerConstraint

Structured constraint specification produced by `_prior_constraints` when
exactly one component of a Kronecker-structured model is constrained. Carries
the materialized sparse constraint system `(A, e)` (block-embedded for
`CombinedModel` composition) alongside the structural decomposition needed
for factor-scale Rue–Held corrections.

# Fields
- `A`: Full sparse constraint matrix (`m × n_total`).
- `e`: Constraint values (`A x = e`).
- `comp`: Index of the constrained factor within its Kronecker node.
- `A_i`: The factor-level constraint block (`m_i × n_i`).
- `block`: Column range of the Kronecker node within the full latent vector
  (the whole vector for a plain `SeparableModel`; the component's block for a
  `CombinedModel`).
"""
struct KroneckerConstraint
    A::SparseMatrixCSC{Float64, Int}
    e::Vector{Float64}
    comp::Int
    A_i::SparseMatrixCSC{Float64, Int}
    block::UnitRange{Int}
end

"""
    _kron_expand_constraint(A_i, e_i, comp, dims) -> (A, e)

Expand a factor-level constraint `(A_i, e_i)` on factor `comp` of a Kronecker
product with factor dimensions `dims` to the full system
`A = I_before ⊗ A_i ⊗ I_after`, with `e` ordered consistently with `A`'s
Kronecker row ordering.
"""
function _kron_expand_constraint(
        A_i::SparseMatrixCSC, e_i::AbstractVector, comp::Int, dims::Vector{Int}
    )
    n_before = prod(dims[1:(comp - 1)]; init = 1)
    n_after = prod(dims[(comp + 1):end]; init = 1)
    A = kron(kron(sparse(I, n_before, n_before), A_i), sparse(I, n_after, n_after))
    m_i = size(A_i, 1)
    # Row of kron(I_b ⊗ A_i, I_a) indexed (b, c, a) is ((b-1)m_i + c - 1)n_a + a
    # and carries e_i[c].
    e = Vector{Float64}(undef, n_before * m_i * n_after)
    r = 0
    for b in 1:n_before, c in 1:m_i, a in 1:n_after
        r += 1
        e[r] = e_i[c]
    end
    return A, e
end

"""
    StructuredPriorConstraints

Constraint state resolved against a concrete structured precision and its
op-cache: the factor matrices and caches of the constrained Kronecker node,
ready for factor-scale correction computations. Built by
`_resolve_constraints` at prior instantiation; only the homogeneous case
(`e == 0`, `μ == 0` — every intrinsic model in practice) is resolved, so the
constrained mean equals the unconstrained mean and the Rue–Held residual
terms vanish.
"""
struct StructuredPriorConstraints{Fs <: Tuple}
    A::SparseMatrixCSC{Float64, Int}
    e::Vector{Float64}
    factors::Fs
    caches::Vector{PrecisionOpCache}
    dims::Vector{Int}
    comp::Int
    A_i::SparseMatrixCSC{Float64, Int}
    block::UnitRange{Int}
end

"""
    _resolve_constraints(Q, cache, kc::KroneckerConstraint) -> StructuredPriorConstraints

Locate the Kronecker node covered by `kc.block` inside the structured
precision `Q` / op-cache tree and bundle its factors and caches with the
constraint specification.
"""
function _resolve_constraints(Q::AbstractKroneckerProduct, cache::StructuredPriorCache, kc::KroneckerConstraint)
    op = cache.op::KroneckerOpCache
    factors = _structure_factors(Q)
    return StructuredPriorConstraints(
        kc.A, kc.e, factors, op.factors, op.dims, kc.comp, kc.A_i, kc.block
    )
end

function _resolve_constraints(Q::BlockDiagonalPrecision, cache::StructuredPriorCache, kc::KroneckerConstraint)
    op = cache.op::BlockDiagOpCache
    for (b, block) in enumerate(Q.blocks)
        rng = (Q.offsets[b] + 1):Q.offsets[b + 1]
        if rng == kc.block
            block isa AbstractKroneckerProduct ||
                throw(ArgumentError("KroneckerConstraint targets a non-Kronecker block."))
            bop = op.blocks[b]::KroneckerOpCache
            return StructuredPriorConstraints(
                kc.A, kc.e, _structure_factors(block), bop.factors, bop.dims,
                kc.comp, kc.A_i, kc.block
            )
        end
    end
    throw(ArgumentError("KroneckerConstraint block $(kc.block) does not match any block of the precision."))
end

# --- Factor-level constraint gram: S = A_i Q_i⁻¹ A_iᵀ (and V = Q_i⁻¹ A_iᵀ) ---

function _constraint_gram(c::SparseOpCache, Q::AbstractMatrix, A_i::SparseMatrixCSC)
    return _constraint_gram_sparse(c.engine, _ensure_sparse(Q), A_i)
end

function _constraint_gram_sparse(
        engine::GMRFWorkspace, Q::SparseMatrixCSC{Float64}, A_i::SparseMatrixCSC
    )
    _same_pattern(Q, engine.Q) ||
        throw(ArgumentError("factor pattern changed since the structured prior cache was built."))
    _leaf_sync!(engine, Q.nzval)
    m_i, n_i = size(A_i)
    V = Matrix{Float64}(undef, n_i, m_i)
    for r in 1:m_i
        V[:, r] = workspace_solve(engine, Vector(A_i[r, :]))
    end
    S = Matrix(A_i * V)
    return S, V
end

# Diagonal factor: Q_i⁻¹ A_iᵀ in closed form. Eltype-generic (exact for Dual).
function _constraint_gram(::DiagonalOpCache, D::Diagonal, A_i::SparseMatrixCSC)
    V = Matrix(A_i') ./ D.diag
    S = Matrix(A_i * V)
    return S, V
end

# --- Rue–Held correction quantities (homogeneous case: resid ≡ 0) ---

_small_logdet(S::AbstractMatrix) =
    size(S, 1) == 1 ? log(S[1, 1]) : logdet(cholesky(Hermitian(Matrix(S))))

"""
    _constraint_logpdf_correction(cs::StructuredPriorConstraints)

The Rue–Held log-density correction (Rue & Held 2005, §2.3.3) for the
homogeneous constrained prior, computed at factor scale:

    0.5 (m log 2π + logdet(A Q⁻¹ Aᵀ)) − 0.5 logdet(A Aᵀ)

with `logdet(A Q⁻¹ Aᵀ) = (m/m_i) logdet(A_i Q_i⁻¹ A_iᵀ) − Σ_{j≠i} (m/n_j) logdet(Q_j)`
and `logdet(A Aᵀ) = (m/m_i) logdet(A_i A_iᵀ)`. The residual terms of the
general formula vanish because `e = 0` and `μ = 0`.
"""
function _constraint_logpdf_correction(cs::StructuredPriorConstraints)
    m = size(cs.A, 1)
    m_i = size(cs.A_i, 1)
    S, _ = _constraint_gram(cs.caches[cs.comp], cs.factors[cs.comp], cs.A_i)
    ld_AQA = (m ÷ m_i) * _small_logdet(S)
    for j in eachindex(cs.dims)
        j == cs.comp && continue
        ld_AQA -= (m ÷ cs.dims[j]) * _structured_logdet(cs.factors[j], cs.caches[j])
    end
    ld_AA = (m ÷ m_i) * _small_logdet(Matrix(cs.A_i * cs.A_i'))
    return 0.5 * (m * log(2π) + ld_AQA) - 0.5 * ld_AA
end

"""
    _constraint_var_correction(cs::StructuredPriorConstraints, n_total) -> Vector

`diag(Q⁻¹Aᵀ (A Q⁻¹ Aᵀ)⁻¹ A Q⁻¹)` — the variance reduction from conditioning
on the constraints — as a length-`n_total` vector (zero outside the
constrained Kronecker node's block). Factorizes as the Kronecker product of
`diag(V S⁻¹ Vᵀ)` on the constrained factor with the selected-inverse
diagonals of the remaining factors.
"""
function _constraint_var_correction(cs::StructuredPriorConstraints, n_total::Int)
    S, V = _constraint_gram(cs.caches[cs.comp], cs.factors[cs.comp], cs.A_i)
    X = V / cholesky(Hermitian(Matrix(S))).U
    w = vec(sum(abs2, X; dims = 2))
    parts = [
        j == cs.comp ? w : _structured_selinv_diag(cs.factors[j], cs.caches[j])
            for j in eachindex(cs.dims)
    ]
    correction = zeros(n_total)
    correction[cs.block] = foldl(kron, parts)
    return correction
end
