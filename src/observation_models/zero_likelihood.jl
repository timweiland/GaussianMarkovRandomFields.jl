export ZeroLikelihood

"""
    ZeroLikelihood <: ObservationLikelihood

A likelihood whose log-density is identically zero — it contributes nothing to the
posterior. This is the *identity* observation term for TMB-style modelling: fold the
entire joint log-density (prior **and** data terms) into an
[`AutoDiffLatentPrior`](@ref), then pair it with `ZeroLikelihood()`. Then
`gaussian_approximation` returns the Laplace approximation of that joint, and
`marginal_loglikelihood` returns the Laplace marginal `log ∫ exp f(x, θ) dx`.

```julia
joint(x; θ...) = log_prior(x; θ...) + log_likelihood(x; θ..., data)   # data captured/baked in
prior = AutoDiffLatentPrior(joint; n, hyperparams)
post  = gaussian_approximation(prior, ZeroLikelihood(); θ...)
logml = marginal_loglikelihood(prior, ZeroLikelihood(), post; θ...)
```

The structured alternative — an `AutoDiffLatentPrior` for the latent part composed
with a separate (possibly exact, closed-form) `ObservationLikelihood` — gives an
identical posterior whenever the joint factorises as prior + likelihood; pick
whichever is more natural to express.
"""
struct ZeroLikelihood <: ObservationLikelihood end

loglik(x, ::ZeroLikelihood) = zero(eltype(x))
loggrad(x, ::ZeroLikelihood) = zeros(eltype(x), length(x))
loghessian(x, ::ZeroLikelihood) = Diagonal(zeros(eltype(x), length(x)))
