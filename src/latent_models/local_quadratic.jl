export LocalLatentQuadratic, local_quadratic

"""
    LocalLatentQuadratic{TQ, Th, T, Tx}

Local quadratic approximation of `log p(x | θ)` around a reference point
`x_ref`, in natural form

    log p(x | θ) ≈ logp_ref + (h - Q · x_ref)' (x - x_ref)
                            - ½ (x - x_ref)' Q (x - x_ref)

equivalently `c + h' x - ½ x' Q x` for an implicit `c`.

# Fields
- `Q::TQ`: `-∇²ₓ log p(x | θ)` evaluated at `x_ref`.
- `h::Th`: `∇ₓ log p(x | θ)|_{x_ref} + Q · x_ref`. The natural-form linear
  coefficient. Well-defined even for rank-deficient (intrinsic) `Q` where
  `μ = Q⁻¹h` is not.
- `logp_ref::T`: Exact `log p(x_ref | θ)` (not the quadratic's value at
  `x_ref`, which agrees only at `x_ref` itself). Carrying the exact value
  lets the Laplace marginal-likelihood formula use it at convergence
  without re-evaluating the prior.
- `x_ref::Tx`: The reference point.
"""
struct LocalLatentQuadratic{TQ, Th, T, Tx}
    Q::TQ
    h::Th
    logp_ref::T
    x_ref::Tx
end

"""
    local_quadratic(m::AbstractLatentPrior, x_ref::AbstractVector; θ...) -> LocalLatentQuadratic

Local quadratic approximation of `log p(x | θ)` at `x_ref` for the latent
prior `m`.

For [`LatentModel`](@ref) (Gaussian) the default method uses
`precision_matrix`, `mean`, and the `logpdf` of the materialised GMRF —
`(Q, h)` are independent of `x_ref` so the Newton iteration reduces to
fixed-Q.

For [`NonGaussianLatentPrior`](@ref) subtypes there is no default; each
concrete subtype must implement this method to specify the natural-form
quadratic at `x_ref`.
"""
function local_quadratic(m::LatentModel, x_ref::AbstractVector; θ...)
    Q = precision_matrix(m; θ...)
    μ = mean(m; θ...)
    h = Q * μ
    logp_ref = logpdf(m(; θ...), x_ref)
    return LocalLatentQuadratic(Q, h, logp_ref, x_ref)
end

function local_quadratic(m::NonGaussianLatentPrior, x_ref::AbstractVector; θ...)
    throw(
        ArgumentError(
            "local_quadratic not implemented for $(typeof(m)). " *
                "Every NonGaussianLatentPrior subtype must implement it."
        )
    )
end

# Internal adapter: bundles an `AbstractLatentPrior` with its
# hyperparameter `NamedTuple` so the Newton loops can dispatch the prior
# side on a single argument.
struct LatentPrior{M <: AbstractLatentPrior, T <: NamedTuple}
    model::M
    θ::T
end

# Per-iterate prior query for the shared Newton loops. Returns the
# natural-form coefficients `(Q, h)` of the local quadratic at `x` plus
# the line-search energy `e = -log p_prior(x)` up to an x-independent
# constant.
#
# For a materialised Gaussian prior `(Q, h)` are independent of `x` and
# the energy is the bare quadratic `½ xᵀQx - hᵀx`. Crucially this never
# calls `logpdf`, so it does NOT trigger a `logdetcov` factorization of a
# shared `GMRFWorkspace` (which stays factorized at `Q_post` for the
# Newton step). For the `LatentPrior` adapter the prior re-linearises per
# iterate and the energy is the *exact* `-log p(x | θ)`.
function _prior_local(prior::AbstractGMRF, x::AbstractVector)
    Q = precision_matrix(prior)
    h = Q * mean(prior)
    return Q, h, 0.5 * dot(x, Q, x) - dot(x, h)
end

function _prior_local(p::LatentPrior, x::AbstractVector)
    lq = local_quadratic(p.model, x; p.θ...)
    return lq.Q, lq.h, -lq.logp_ref
end

# Line-search energy at an arbitrary point. Gaussian priors reuse the
# at-iterate `(Q, h)` (constant in `x`); the `LatentPrior` adapter
# re-evaluates the exact prior log-density.
_prior_energy(::AbstractGMRF, Q, h, x::AbstractVector) = 0.5 * dot(x, Q, x) - dot(x, h)
_prior_energy(p::LatentPrior, _Q, _h, x::AbstractVector) =
    -local_quadratic(p.model, x; p.θ...).logp_ref
