export linear_predictor_marginals

"""
    linear_predictor_marginals(ga, obs_lik) -> (μ_η, v_η, eta_likelihood)

Posterior marginals of the per-observation linear predictor `η_i` under a
Gaussian approximation `ga` to the latent posterior.

For each observation `i` returns `(μ_η[i], v_η[i])` — the mean and variance of
the scalar predictor `η_i` that `obs_lik` consumes. The third return value is
the observation likelihood with any wrapping that mediates between `x` and `η`
stripped away, so consumers can evaluate `p(y | η)` directly without going
back through the design matrix.

The dispatch recurses on the observation likelihood structure:

- `ExponentialFamilyLikelihood`: `η_i = x[indices[i]]` (or `η_i = x_i` when
  `indices === nothing`), so the result is `(mean(ga)[idx], var(ga)[idx], lik)`.
- `LinearlyTransformedLikelihood`: `η = A x`, giving
  `μ_η = A · mean(ga)` and `v_η = diag(A · Σ · Aᵀ)` where `Σ` comes from the
  posterior's selected-inversion output. The third return is the stripped
  `lik.base_likelihood`.
- `CompositeLikelihood`: per-component results concatenated. The third return
  is the original composite likelihood — consumers wanting per-observation
  `p(y_i | η_i)` evaluation should slice `(μ_η, v_η)` per component.

# Hard constraints

If `ga` carries a hard linear constraint `A_c x = e` (either a `ConstrainedGMRF`
or a `WorkspaceGMRF` with constraint info), `v_η` includes the constraint
correction `diag(A · A_tilde_T · L_c⁻ᵀ · L_c⁻¹ · A_tilde_Tᵀ · Aᵀ)` subtracted
from the unconstrained `diag(A · Σ · Aᵀ)`, using the cached `A_tilde_T =
Σ A_cᵀ` and `L_c = chol(A_c Σ A_cᵀ)`. The mean is the constrained mean
returned by `mean(ga)`.

# Sparse-pattern assumption

For `LinearlyTransformedLikelihood`, the variance computation reads `Σ` from
the posterior's selected inversion, which fills only entries at the Cholesky
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
    return (Vector{Float64}(μ_full[idx]), Vector{Float64}(v_full[idx]), lik)
end

function linear_predictor_marginals(ga::AbstractGMRF, lik::LinearlyTransformedLikelihood)
    A = lik.design_matrix
    μ_η = A * mean(ga)
    v_η = _row_diag_AΣAt(A, ga)
    _apply_lpm_constraint_correction!(v_η, A, ga)
    return (Vector{Float64}(μ_η), v_η, lik.base_likelihood)
end

function linear_predictor_marginals(ga::AbstractGMRF, lik::CompositeLikelihood)
    parts = map(c -> linear_predictor_marginals(ga, c), lik.components)
    μ_η = reduce(vcat, getindex.(parts, 1))
    v_η = reduce(vcat, getindex.(parts, 2))
    return (Vector{Float64}(μ_η), Vector{Float64}(v_η), lik)
end

_posterior_cov_sparse(ga::WorkspaceGMRF) = selinv(ga.workspace)
_posterior_cov_sparse(ga::GMRF) = selinv(ga.linsolve_cache)
_posterior_cov_sparse(ga::ConstrainedGMRF) = _posterior_cov_sparse(ga.base_gmrf)

# `_row_diag_AΣAt` reads the *unconstrained* selinv. The constraint correction
# below subtracts diag(A · Σ A_c' · (A_c Σ A_c')⁻¹ · A_c Σ · Aᵀ), reusing the
# cached `A_tilde_T = Σ A_c'` and `L_c = chol(A_c Σ A_c')` on the constraint
# info. Mirrors what `var(::WorkspaceGMRF)` / `var(::ConstrainedGMRF)` do for
# the per-coordinate diagonal, generalised to the A-row contraction.
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

# v[i] = sum_{j,k} A[i,j] · A[i,k] · Σ[j,k]
#      = (AΣ * Aᵀ)[i, i]
#      = elementwise dot of row i of A and row i of AΣ.
# Sparse arithmetic handles the pattern intersection; entries of Σ outside its
# stored pattern contribute zero, which is exact when the pattern assumption
# in the docstring holds.
function _row_diag_AΣAt(A::AbstractMatrix, ga::AbstractGMRF)
    Σ = _posterior_cov_sparse(ga)
    AΣ = A * Σ
    return Vector{Float64}(vec(sum(AΣ .* A, dims = 2)))
end
