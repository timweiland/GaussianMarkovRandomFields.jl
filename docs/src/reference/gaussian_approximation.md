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

## API Reference

```@docs
gaussian_approximation
```
