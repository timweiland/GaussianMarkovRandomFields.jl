import Distributions: logpdf
using ChainRulesCore
using Random

export StructuredPriorGMRF

"""
    StructuredPriorGMRF{T, QS, W} <: AbstractGMRF

A Gaussian latent prior whose precision is carried in structured form
(Kronecker product / block diagonal) alongside a sparse snapshot on the joint
workspace pattern. Returned by `(model::LatentModel)(ws::GMRFWorkspace; θ...)`
when the model's `precision_matrix` is structured and the model is
unconstrained.

Division of labor:

- **Scalar queries** (`logdetcov`, `logpdf`, `var`) are answered from
  structure, via small per-factor engines in the attached
  [`StructuredPriorCache`](@ref) — the joint workspace is *never* factorized
  at the prior. `logdet(⊗ᵢ Qᵢ) = Σᵢ (N/nᵢ) logdet(Qᵢ)`.
- **The posterior update** consumes the sparse `precision` snapshot (values
  on the workspace pattern) positionally in the Newton loop; the joint
  workspace's single factor slot holds only `Q_post`.

Consequently, unlike a `WorkspaceGMRF` prior, a `StructuredPriorGMRF` never
competes with the posterior for the joint factorization — the
prior-vs-posterior refactorization thrash cannot occur by construction.

# Fields
- `mean`: Prior mean.
- `structure`: The lazy structured precision (`KroneckerProduct` /
  [`BlockDiagonalPrecision`](@ref)).
- `precision`: Sparse snapshot of the same values on the workspace pattern
  (shared `colptr`/`rowval` with `workspace.Q`).
- `workspace`: The joint `GMRFWorkspace` — used only by
  `gaussian_approximation` to run the posterior Newton loop.
- `cache`: Per-factor engines + scatter map (see `StructuredPriorCache`).
"""
struct StructuredPriorGMRF{
        T <: Real, QS <: AbstractMatrix, W <: GMRFWorkspace,
    } <: AbstractGMRF{T, SparseMatrixCSC{T, Int}}
    mean::Vector{T}
    structure::QS
    precision::SparseMatrixCSC{T, Int}
    workspace::W
    cache::StructuredPriorCache
end

function StructuredPriorGMRF(
        mean::AbstractVector, structure::AbstractMatrix,
        precision::SparseMatrixCSC, ws::GMRFWorkspace, cache::StructuredPriorCache
    )
    T = promote_type(eltype(mean), eltype(precision))
    mean_T = Vector{T}(mean)
    precision_T = eltype(precision) === T ? precision :
        SparseMatrixCSC(
            precision.m, precision.n, precision.colptr, precision.rowval,
            convert(Vector{T}, precision.nzval)
        )
    return StructuredPriorGMRF{T, typeof(structure), typeof(ws)}(
        mean_T, structure, precision_T, ws, cache
    )
end

# --- AbstractGMRF interface ---

Base.length(d::StructuredPriorGMRF) = size(d.precision, 1)
mean(d::StructuredPriorGMRF) = d.mean
precision_map(d::StructuredPriorGMRF) = d.precision
precision_matrix(d::StructuredPriorGMRF) = d.precision

"""
    logdetcov(d::StructuredPriorGMRF)

`log |Q⁻¹| = -log |Q|`, computed from structure (factor-level factorizations
via the cached engines). Never touches the joint workspace.
"""
logdetcov(d::StructuredPriorGMRF) = -_structured_logdet(d.structure, d.cache.op)

function logpdf(d::StructuredPriorGMRF, z::AbstractVector)
    r = z - d.mean
    n = length(d)
    return -0.5 * dot(r, d.precision, r) - 0.5 * logdetcov(d) - 0.5 * n * log(2π)
end

var(d::StructuredPriorGMRF) = _structured_selinv_diag(d.structure, d.cache.op)

std(d::StructuredPriorGMRF) = sqrt.(var(d))

function _rand!(rng::AbstractRNG, d::StructuredPriorGMRF, x::AbstractVector)
    throw(
        ArgumentError(
            "Sampling from a StructuredPriorGMRF is not yet implemented. " *
                "Materialize the prior for sampling, e.g. " *
                "`GMRF(mean(d), sparse(d.precision))`."
        )
    )
end

# --- Reverse-mode guards ---
#
# Reverse-mode AD support for the structured prior is not implemented yet.
# Without explicit rules, ChainRules-based engines (Zygote) would trace into
# factor-engine internals and — worst case — silently drop the prior
# log-determinant term (a NoTangent factorization field zeroes the whole
# term; see the ChordalGMRF logdetcov incident). Fail loudly instead.

const _STRUCTURED_REVERSE_AD_MSG =
    "Reverse-mode AD through a StructuredPriorGMRF is not supported yet. " *
    "Use ForwardDiff for hyperparameter gradients (exact for this path), or " *
    "construct the prior from a materialized sparse precision."

function ChainRulesCore.rrule(::typeof(logdetcov), d::StructuredPriorGMRF)
    throw(ArgumentError(_STRUCTURED_REVERSE_AD_MSG))
end

function ChainRulesCore.rrule(::typeof(logpdf), d::StructuredPriorGMRF, z::AbstractVector)
    throw(ArgumentError(_STRUCTURED_REVERSE_AD_MSG))
end

# --- Display ---

function Base.show(io::IO, d::StructuredPriorGMRF{T}) where {T}
    return print(io, "StructuredPriorGMRF{$T}(n=$(length(d)), structure=$(_structure_summary(d.structure)))")
end

_structure_summary(K::AbstractKroneckerProduct) =
    join(string.(size.(_structure_factors(K), 1)), " ⊗ ")
_structure_summary(B::BlockDiagonalPrecision) =
    "blockdiag(" * join(_structure_summary.(B.blocks), ", ") * ")"
_structure_summary(A::AbstractMatrix) = string(size(A, 1))

# COV_EXCL_START
function Base.show(io::IO, ::MIME"text/plain", d::StructuredPriorGMRF{T}) where {T}
    println(io, "StructuredPriorGMRF{$T} with $(length(d)) variables")
    println(io, "  Structure: $(_structure_summary(d.structure))")
    return print(io, "  Backend: $(typeof(d.workspace.backend))")
end
# COV_EXCL_STOP
