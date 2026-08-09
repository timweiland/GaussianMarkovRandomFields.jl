# Gaussian Approximation

When using non-Gaussian observation models (Poisson, Bernoulli, etc.) with GMRF priors, the posterior distribution p(x|y) is no longer Gaussian. Gaussian approximation finds the "best" Gaussian distribution that approximates this intractable posterior, enabling efficient inference using the full power of GMRF machinery.

The `gaussian_approximation` function constructs this approximation by finding the posterior mode and using the curvature at the mode to define a Gaussian with matching first and second moments. This unlocks fast sampling, marginal variance computation, and further conditioning operations.

## Conjugate vs Non-Conjugate Cases

The package automatically detects two fundamentally different cases:

**Conjugate Case (Exact)**: Normal observations with identity link result in a Gaussian posterior that can be computed exactly using linear conditioning. No approximation is needed.

**Non-Conjugate Case (Approximate)**: All other observation models require iterative optimization using Fisher scoring to find the mode and Hessian of the log-posterior.

## Basic Usage Pattern

```julia
# Step 1: Set up prior GMRF
prior_gmrf = GMRF(μ_prior, Q_prior)

# Step 2: Set up observation model and materialize likelihood
obs_model = ExponentialFamily(Poisson)  # or Normal, Bernoulli, etc.
obs_lik = obs_model(y_data; hyperparameters...)

# Step 3: Find Gaussian approximation to posterior
posterior_gmrf = gaussian_approximation(prior_gmrf, obs_lik)

# Step 4: Use like any other GMRF
sample = rand(posterior_gmrf)
posterior_mean = mean(posterior_gmrf)
marginal_stds = std(posterior_gmrf)
```

## Examples

### Conjugate Case: Normal Observations

For Normal observations with identity link, the posterior is exactly Gaussian:

```julia
using GaussianMarkovRandomFields, Distributions, LinearAlgebra

# Prior: zero mean, unit precision
n = 10
prior_gmrf = GMRF(zeros(n), Diagonal(ones(n)))

# Normal observations: y ~ N(x, σ²I) 
obs_model = ExponentialFamily(Normal)
y = randn(n)  # Some observed data
obs_lik = obs_model(y; σ=0.5)

# Exact posterior (no iteration needed!)
posterior_gmrf = gaussian_approximation(prior_gmrf, obs_lik)
```

This is mathematically equivalent to Bayesian linear regression and is computed exactly using the conjugate prior relationship.

### Non-Conjugate Case: Poisson GLM

For count data with log-link, we get an approximate Gaussian posterior:

```julia
# Prior for log-intensities
prior_gmrf = GMRF(zeros(n), Diagonal(ones(n)))

# Poisson observations: y ~ Poisson(exp(x))
obs_model = ExponentialFamily(Poisson)  # Uses LogLink by default
y_counts = [3, 1, 4, 1, 5, 2, 6, 3, 5, 4]  # Count data
obs_lik = obs_model(y_counts)

# Approximate Gaussian posterior via Fisher scoring
posterior_gmrf = gaussian_approximation(prior_gmrf, obs_lik)
```

The approximation quality depends on how "Gaussian-like" the true posterior is. For moderate counts and reasonable prior precision, the approximation is typically excellent.

### Design Matrices: Still Conjugate

Even with linear transformations, Normal observations remain conjugate:

```julia
# GLM-style design matrix: intercept + covariate
X = [1.0 2.1; 1.0 3.4; 1.0 1.8; 1.0 4.2]  # 4 obs, 2 coefficients
n_coef = 2

# Prior on coefficients [β₀, β₁]
prior_gmrf = GMRF(zeros(n_coef), Diagonal(ones(n_coef)))

# Normal observations with design matrix
base_model = ExponentialFamily(Normal)
obs_model = LinearlyTransformedObservationModel(base_model, X)
y = [2.3, 3.8, 2.0, 4.5]  # Response data
obs_lik = obs_model(y; σ=0.3)

# Still exact! Uses linear conditioning internally
posterior_gmrf = gaussian_approximation(prior_gmrf, obs_lik)
```

This covers standard GLM scenarios while maintaining computational efficiency.

## Performance Notes

The package includes significant performance optimizations:

- **Conjugate cases** are detected automatically and use exact linear conditioning instead of iterative optimization, providing both speed and numerical accuracy benefits
- **Non-conjugate cases** use efficient Fisher scoring with good initial guesses from the prior mean
- **MetaGMRF support** preserves metadata through the approximation process
- **Type consistency** ensures optimal memory usage and dispatch efficiency

### Step-size line search

Each Newton iteration costs a factorization plus a solve; the backtracking line search that
guards it costs only merit evaluations (`½xᵀQx - hᵀx` for the prior side plus `loglik`, never
a `logpdf`, hence never a factorization). The step scale `α` persists across iterations —
a rejected step shrinks it by 10× — so the policy that grows it back decides how many
*damped* (but fully factorized) iterations a single early backtrack costs:

- `step_recovery = :retry_full` (default) probes the undamped step `α = 1` at the start of
  every iteration and falls back to the persistent damped scale only if that step is
  rejected. Cost: one extra merit evaluation on the iterations where the full step is not
  yet acceptable. Benefit: full Newton steps resume the moment the iterate reaches the basin
  where they are accepted.
- `step_recovery = :sqrt` grows `α` as `sqrt(α)` per accepted step. A backtrack down to
  `α = 0.1` then takes ~8–10 further iterations to recover, each of them a needlessly damped
  factorize + solve. Use this if the merit is not convex along the Newton direction (e.g. a
  [`NonGaussianLatentPrior`](@ref) whose log-density is not concave) and the slow recovery is
  wanted as an oscillation guard.

For canonical-link exponential families the merit is convex along the Newton direction, so
the default is both faster and — because it stops the mean-change convergence test from
firing on an artificially damped step — at least as accurate.

A step is accepted when it does not *significantly* increase the merit. The tolerance is
the merit's own rounding-noise floor (a few dozen ULPs of the magnitude of the terms
summed to form it). It matters only on the convergence plateau: there the true decrease is
smaller than the noise in evaluating it, so a strict decrease test rejects a perfectly good
undamped Newton step on a 1–2 ULP artifact and substitutes a 10× damped one. That put a
floor under the attainable mode — `‖∇ₓ neg-log-posterior‖∞ ≈ 2e-10` on a 200-variable
Poisson chain, versus `7e-15` with the tolerance — and meant that re-running
`gaussian_approximation` from an already-converged mode moved away from it instead of
staying put.

## Iterated linearisation for non-Gaussian priors

`gaussian_approximation` also accepts an [`AbstractLatentPrior`](@ref) directly as the prior side, which is useful in two situations:

1. **Gaussian latent prior + hyperparameters in scope.** Pass the [`LatentModel`](@ref) (e.g. `AR1Model(n)`) and its hyperparameters as keyword arguments — the call materialises the prior internally and dispatches to the existing `(::AbstractGMRF, obs_lik)` path. Behaviour is identical to first calling `model(; θ...)` and then `gaussian_approximation(prior_gmrf, obs_lik)`.

2. **Non-Gaussian latent prior.** When the prior log-density is not quadratic in `x` (a [`NonGaussianLatentPrior`](@ref) — e.g. a nonlinear state-space model), fixed-Q Newton on a single Gaussian approximation is biased: the prior Hessian depends on `x`, so the linearisation has to track the current Newton iterate. The dispatch on `NonGaussianLatentPrior` runs *iterated re-linearisation* — at every Newton step the prior is re-quadratised at the current iterate via [`local_quadratic`](@ref), and the joint Newton system is solved against the re-linearised prior. The line-search merit uses the *exact* `log p(x | θ)`, evaluated at trial points through [`prior_logdensity`](@ref) (which defaults to `local_quadratic(prior, x; θ...).logp_ref` but can be overridden to evaluate the scalar log-density directly, avoiding a full Hessian rebuild per backtrack).

Both cases share the same Newton machinery (cache-backed via `LinearSolve` or workspace-backed via `GMRFWorkspace`); the prior side is only queried via a per-iterate local-quadratic hook (returning `(Q, h)` plus a line-search energy), so the loop body is identical regardless of whether the prior is Gaussian. For Gaussian priors that hook never evaluates `logpdf`, so the line search adds no factorization on a shared workspace.

```julia
using GaussianMarkovRandomFields

# Define a non-Gaussian latent prior — only requires `local_quadratic`
# plus the AbstractLatentPrior interface (length, hyperparameters,
# model_name, constraints).
struct MyNonlinearPrior <: NonGaussianLatentPrior
    n::Int
end

Base.length(m::MyNonlinearPrior) = m.n
GaussianMarkovRandomFields.hyperparameters(::MyNonlinearPrior) = (τ = Real,)
GaussianMarkovRandomFields.model_name(::MyNonlinearPrior) = :my_nonlinear
GaussianMarkovRandomFields.constraints(::MyNonlinearPrior; kwargs...) = nothing

function GaussianMarkovRandomFields.local_quadratic(
        m::MyNonlinearPrior, x_ref::AbstractVector; τ::Real
    )
    # Compute Q = -∇²log p(x_ref | τ), the natural-form linear coefficient
    # h = ∇log p(x_ref) + Q · x_ref, and the *exact* log p(x_ref | τ).
    Q = ...
    h = ...
    logp_ref = ...
    return LocalLatentQuadratic(Q, h, logp_ref, x_ref)
end

# Use it.
model = MyNonlinearPrior(50)
posterior = gaussian_approximation(model, obs_lik; τ = 1.0, x0 = my_initial_guess)
logml = marginal_loglikelihood(model, obs_lik, posterior; τ = 1.0)
```

### Pitfall: symmetric saddles

The default initial point for `NonGaussianLatentPrior` is `zeros(length(prior))`. This is the worst possible start for any prior with a discrete reflection symmetry — Newton from `x = 0` stays on the symmetry axis because the gradient identically vanishes there. For symmetric models (e.g. transitions involving `x²`, `|x|`, etc.), pass `x0` explicitly:

```julia
posterior = gaussian_approximation(
    model, obs_lik;
    τ = 1.0,
    x0 = fill(0.5, length(model)),   # any non-zero start works
)
```

A common pattern for state-space models is to initialise from the data (e.g. `x0 = y` for direct-observation likelihoods) or from the mode of a simpler Gaussian-prior approximation.

### Marginal log-likelihood

[`marginal_loglikelihood`](@ref) returns the Laplace approximation to `log p(y | θ)` at the converged mode. It uses the elementary identity `log p(y | θ) = log p(x*, y | θ) - log p(x* | y, θ)`, which handles unconstrained and constrained posteriors uniformly via the posterior's `Distributions.logpdf`.

## API Reference

```@docs
gaussian_approximation
marginal_loglikelihood
```
