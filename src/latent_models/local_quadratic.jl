export LocalLatentQuadratic, local_quadratic, prior_quadratic

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
    return error("local_quadratic not implemented for $(typeof(m))")
end

# Internal adapter: bundles an `AbstractLatentPrior` with its
# hyperparameter `NamedTuple` so the Newton loops can dispatch on a
# single argument.
struct LatentPrior{M <: AbstractLatentPrior, T <: NamedTuple}
    model::M
    θ::T
end

prior_quadratic(p::LatentPrior, x_ref::AbstractVector) =
    local_quadratic(p.model, x_ref; p.θ...)

function prior_quadratic(prior::AbstractGMRF, x_ref::AbstractVector)
    Q = precision_matrix(prior)
    return LocalLatentQuadratic(Q, Q * mean(prior), logpdf(prior, x_ref), x_ref)
end
