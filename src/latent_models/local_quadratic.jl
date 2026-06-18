export LocalLatentQuadratic, local_quadratic, prior_logdensity

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

"""
    prior_logdensity(m::AbstractLatentPrior, x::AbstractVector; θ...) -> Real

Exact `log p(x | θ)` of the latent prior at `x`.

The iterated-linearisation line search evaluates the prior log-density at trial
points where it needs only this scalar — not the full local quadratic. The
default delegates to `local_quadratic(m, x; θ...).logp_ref`, so models that
don't override it keep working unchanged. A [`NonGaussianLatentPrior`](@ref)
whose `local_quadratic` is expensive (e.g. it assembles a sparse Hessian) can
override `prior_logdensity` with a direct, allocation-light evaluation of
`log p(x | θ)` to make the line search cheaper.
"""
prior_logdensity(m::AbstractLatentPrior, x::AbstractVector; θ...) =
    local_quadratic(m, x; θ...).logp_ref

# --- IFT hooks (hyperparameter-gradient path) ---
# The Implicit Function Theorem path needs the prior's latent gradient and Hessian evaluated at
# a `ForwardDiff.Dual`-valued `x` (and Dual `θ`). Routing these through hooks — rather than
# hard-coding `DI.gradient`/`DI.hessian` over an opaque `logp_func` — lets a structured
# (factor-graph) prior assemble them from small per-factor derivatives instead of differentiating
# the whole-model closure (which is what makes the per-model compile cost large).

"""
    _dual_prior_gradient(m::NonGaussianLatentPrior, x_dual, θ::NamedTuple) -> ∇ₓ log p(x | θ)

IFT hook: latent-gradient of the prior log-density at a Dual-valued `x_dual` (Dual `θ`), forming
the score's θ-tangent. Must nest cleanly under an outer Dual (forward-mode AD).
"""
function _dual_prior_gradient end

"""
    _dual_prior_hessian(m::NonGaussianLatentPrior, x_dual, x_primal, θ::NamedTuple, θ_primal::NamedTuple) -> ∇²ₓ log p

IFT hook: latent-Hessian of the prior log-density at the Dual-valued `x_dual`, returned sparse
with the prior's structural pattern (detected at the primal `x_primal`/`θ_primal`). Builds the
Dual posterior precision in the IFT path.
"""
function _dual_prior_hessian end

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
# re-evaluates the exact prior log-density via `prior_logdensity`, which a
# subtype can override to avoid rebuilding the full local quadratic per
# trial point.
_prior_energy(::AbstractGMRF, Q, h, x::AbstractVector) = 0.5 * dot(x, Q, x) - dot(x, h)
_prior_energy(p::LatentPrior, _Q, _h, x::AbstractVector) = -prior_logdensity(p.model, x; p.θ...)
