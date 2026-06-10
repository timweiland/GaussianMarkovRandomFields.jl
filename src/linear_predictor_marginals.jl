export linear_predictor_marginals

"""
    linear_predictor_marginals(ga, obs_lik) -> (μ_η, v_η, eta_likelihood)

Posterior marginals of the per-observation linear predictor `η_i` under a
Gaussian approximation `ga` to the latent posterior.

For each observation `i` the result contains `(μ_η[i], v_η[i])` — the mean and
variance of the scalar predictor `η_i` that `obs_lik` consumes. The third
return value is an observation likelihood whose own indexing matches `μ_η`'s
layout: feeding `μ_η` directly into `loglik`, `pointwise_loglik`, `loggrad`,
or `loghessian` yields per-observation outputs in the same order as `μ_η`.

The dispatch recurses on the observation likelihood structure:

- `ExponentialFamilyLikelihood`: `η_i = x[indices[i]]` (or `η_i = x_i` when
  `indices === nothing`). `μ_η` and `v_η` are the corresponding slices of
  `mean(ga)` / `var(ga)`; `eta_likelihood` is the same likelihood with
  `indices === nothing` so it consumes the returned (smaller) `μ_η` directly.
- `LinearlyTransformedLikelihood`: affine predictor `η = A x + b` (the offset
  `b = lik.offset` is omitted when `nothing`), giving `μ_η = A · mean(ga) + b`
  and `v_η = diag(A · Σ · Aᵀ)` from the posterior's selected-inversion output
  (the constant `b` does not affect the variance).
  `eta_likelihood = lik.base_likelihood`. Assumes the base's own `indices`
  field is `nothing` (the standard wrapping pattern); an indexed base is
  unusual and not specially handled.
- `CompositeLikelihood`: per-component results concatenated. `eta_likelihood`
  is a fresh `CompositeLikelihood` whose components are the per-component
  stripped likelihoods with `indices` re-assigned to their slice of the
  concatenated `η`, so the result can be evaluated against `μ_η` directly.
  Composite-of-composite is not specially handled — flatten upstream.

# Hard constraints

If `ga` carries a hard linear constraint `A_c x = e` (either a
`ConstrainedGMRF` or a `WorkspaceGMRF` with constraint info populated), `v_η`
subtracts the standard correction
`diag(A · A_tilde_T · L_c⁻ᵀ · L_c⁻¹ · A_tilde_Tᵀ · Aᵀ)` from the unconstrained
`diag(A · Σ · Aᵀ)`, reusing the cached `A_tilde_T = Σ A_cᵀ` and
`L_c = chol(A_c Σ A_cᵀ)`. The mean is the constrained mean returned by
`mean(ga)`.

# Sparse-pattern assumption

For `LinearlyTransformedLikelihood`, the variance computation reads `Σ` from
the posterior's selected inversion, which only fills entries at the Cholesky
factor pattern of the precision matrix `Q`. When `Q`'s pattern subsumes that
of `Aᵀ A` — automatic when the posterior comes out of `gaussian_approximation`
with a `LinearlyTransformedObservationModel` obs side — every entry `Σ[j, k]`
needed by `diag(A Σ Aᵀ)` is present and the result is exact. With a
hand-rolled prior whose pattern is too narrow, missing `Σ` entries silently
contribute zero and `v_η` underestimates; arrange `Q`'s pattern to include
`Aᵀ A` if you build the posterior outside the package's standard flow.
"""
function linear_predictor_marginals end

function linear_predictor_marginals(ga::AbstractGMRF, lik::ExponentialFamilyLikelihood)
    μ_full = mean(ga)
    v_full = var(ga)
    idx = lik.indices
    if idx === nothing
        return (Vector{Float64}(μ_full), Vector{Float64}(v_full), lik)
    end
    return (
        Vector{Float64}(μ_full[idx]),
        Vector{Float64}(v_full[idx]),
        _with_indices(lik, nothing),
    )
end

function linear_predictor_marginals(ga::AbstractGMRF, lik::LinearlyTransformedLikelihood)
    A = lik.design_matrix
    μ_η = A * mean(ga)
    # Affine predictor η = A·x + b: include the additive offset in the mean.
    # (The constant b does not affect the variance.)
    lik.offset !== nothing && (μ_η = μ_η .+ lik.offset)
    v_η = _row_diag_AΣAt(A, ga)
    _apply_lpm_constraint_correction!(v_η, A, ga)
    return (Vector{Float64}(μ_η), v_η, lik.base_likelihood)
end

function linear_predictor_marginals(ga::AbstractGMRF, lik::CompositeLikelihood)
    parts = map(c -> linear_predictor_marginals(ga, c), lik.components)
    μ_η = reduce(vcat, getindex.(parts, 1))
    v_η = reduce(vcat, getindex.(parts, 2))

    offsets = cumsum([length(p[1]) for p in parts])
    reindexed = ntuple(length(parts)) do i
        lo = i == 1 ? 1 : offsets[i - 1] + 1
        hi = offsets[i]
        _with_indices(parts[i][3], lo:hi)
    end
    return (Vector{Float64}(μ_η), Vector{Float64}(v_η), CompositeLikelihood(reindexed))
end

# Rebuild a materialised ExponentialFamilyLikelihood with a different `indices`
# field. Kept narrow to the seven concrete subtypes — only the post-fit
# primitive needs this seam, so promoting it to a package-wide trait would be
# premature.
_with_indices(lik::NormalLikelihood, idx) =
    NormalLikelihood(lik.link, lik.y, lik.σ, lik.inv_σ², lik.log_σ, idx)
_with_indices(lik::PoissonLikelihood, idx) =
    PoissonLikelihood(lik.link, lik.y, idx, lik.logexposure)
_with_indices(lik::BernoulliLikelihood, idx) =
    BernoulliLikelihood(lik.link, lik.y, idx)
_with_indices(lik::BinomialLikelihood, idx) =
    BinomialLikelihood(lik.link, lik.y, lik.n, idx)
_with_indices(lik::NegBinLikelihood, idx) =
    NegBinLikelihood(lik.link, lik.y, lik.r, idx, lik.logexposure)
_with_indices(lik::GammaLikelihood, idx) =
    GammaLikelihood(lik.link, lik.y, lik.phi, idx)
_with_indices(lik::StudentTLikelihood, idx) =
    StudentTLikelihood(lik.link, lik.y, lik.σ, lik.ν, lik.w, lik.νp1, lik.σ_eff, idx)

_posterior_cov_sparse(ga::WorkspaceGMRF) = selinv(ga.workspace)
_posterior_cov_sparse(ga::GMRF) = selinv(ga.linsolve_cache)
_posterior_cov_sparse(ga::ConstrainedGMRF) = _posterior_cov_sparse(ga.base_gmrf)

# v[i] = sum_{j,k} A[i,j] · A[i,k] · Σ[j,k] = (A Σ Aᵀ)[i, i]. Only the diagonal is
# needed, and it depends only on the observation-local blocks of Σ. `Σ` is the
# selected inverse, so `Σ[j, k]` is the true covariance for stored entries and zero
# outside the factor's fill pattern (the documented sparse-pattern assumption); the
# rewrite below preserves that behavior exactly.
function _row_diag_AΣAt(A::AbstractMatrix, ga::AbstractGMRF)
    Σ = _posterior_cov_sparse(ga)
    return _row_diag_AΣAt(A, Σ)
end

# Fast path (#159): for a sparse design matrix, read only each observation row's
# local block of Σ — `O(m · nnz_per_row²)` instead of forming the full `m × n`
# product `A * Σ` (which costs `O(m · nnz(Σ)/row)`, growing with the Cholesky fill).
# Dispatch keys on `A::SparseMatrixCSC` and leaves `Σ` untyped on purpose: on the
# GMRF path `Σ` is a `Symmetric` wrapper (whose `getindex` reflects both triangles),
# on the workspace path a plain `SparseMatrixCSC`; indexing `Σ[j, k]` directly
# reproduces the old `A * Σ` result exactly for every backend (and 0 outside the
# stored pattern). Do NOT unwrap `Σ.data` — that would read only one bare triangle.
function _row_diag_AΣAt(A::SparseMatrixCSC, Σ::AbstractMatrix)
    At = sparse(transpose(A))
    rv = rowvals(At)
    nz = nonzeros(At)
    out = zeros(Float64, size(A, 1))
    @inbounds for i in eachindex(out)
        rng = nzrange(At, i)
        s = 0.0
        for p in rng, q in rng
            s += Float64(nz[p] * nz[q] * Σ[rv[p], rv[q]])
        end
        out[i] = s
    end
    return out
end

# Generic fallback: dense (or otherwise non-`SparseMatrixCSC`) design matrices have
# per-row support ≈ n, where forming the BLAS-backed `A * Σ` product is the right
# choice. This is the original implementation.
function _row_diag_AΣAt(A::AbstractMatrix, Σ::AbstractMatrix)
    AΣ = A * Σ
    return Vector{Float64}(vec(sum(AΣ .* A, dims = 2)))
end

# `_row_diag_AΣAt` reads the *unconstrained* selinv; subtract the constraint
# correction here, reusing the cached `A_tilde_T` / `L_c`. Mirrors what
# `var(::WorkspaceGMRF)` / `var(::ConstrainedGMRF)` do for the per-coordinate
# diagonal, generalised to the A-row contraction.
_apply_lpm_constraint_correction!(v_η, A, ga::GMRF) = v_η
function _apply_lpm_constraint_correction!(v_η, A, ga::WorkspaceGMRF{T}) where {T}
    ci = ga.constraints
    ci === nothing && return v_η
    return _subtract_constraint_correction!(v_η, A, ci.A_tilde_T, ci.L_c, T)
end
function _apply_lpm_constraint_correction!(v_η, A, ga::ConstrainedGMRF{T}) where {T}
    return _subtract_constraint_correction!(v_η, A, ga.A_tilde_T, ga.L_c, T)
end

function _subtract_constraint_correction!(v_η, A, A_tilde_T, L_c, ::Type{T}) where {T}
    M = A * A_tilde_T                          # m × r
    B_T = L_c.L \ M'                           # r × m
    v_η .-= vec(sum(abs2, B_T, dims = 1))
    v_η .= max.(v_η, zero(T))
    return v_η
end
