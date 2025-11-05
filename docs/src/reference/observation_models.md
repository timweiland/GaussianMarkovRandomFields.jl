# Observation Models

Observation models define the relationship between observations `y` and the latent GMRF field `x`, typically through likelihood functions. They enable Bayesian inference by connecting your data to the underlying Gaussian process through flexible probabilistic models.

GaussianMarkovRandomFields.jl implements observation models using a **factory pattern** that separates model configuration from materialized evaluation instances. This design provides major performance benefits in optimization loops and cleaner automatic differentiation boundaries.

## Core Concepts

### The Factory Pattern

Observation models follow a two-stage pattern:

1. **ObservationModel**: A factory that defines the model structure and hyperparameters
2. **ObservationLikelihood**: A materialized instance with specific data and hyperparameters for fast evaluation

```julia
# Step 1: Configure observation model (factory)
obs_model = ExponentialFamily(Normal)

# Step 2: Materialize with data and hyperparameters  
obs_lik = obs_model(y; σ=1.2)

# Step 3: Fast evaluation in hot loops
ll = loglik(x, obs_lik)      # Only x argument needed!
grad = loggrad(x, obs_lik)   # Fast x-only evaluation
hess = loghessian(x, obs_lik)
```

This pattern eliminates the need to repeatedly pass data and hyperparameters, providing significant performance benefits in optimization and sampling algorithms.

### Evaluation Interface

All materialized observation likelihoods support a common interface:

- `loglik(x, obs_lik)`: Evaluate log-likelihood
- `loggrad(x, obs_lik)`: Compute gradient with respect to latent field
- `loghessian(x, obs_lik)`: Compute Hessian matrix
- `pointwise_loglik(x, obs_lik)`: Compute per-observation log-likelihoods (see below)
- `pointwise_loglik!(result, x, obs_lik)`: In-place version

### Pointwise Log-Likelihoods

For model comparison metrics like WAIC, LOO-CV, and CPO, you need per-observation log-likelihoods rather than just the total. The `pointwise_loglik` function returns a vector where each element is the log-likelihood of one observation:

```julia
obs_model = ExponentialFamily(Poisson)
obs_lik = obs_model([1, 3, 0, 2])
x = [0.5, 1.2, -0.3, 0.8]

# Total log-likelihood (scalar)
total_ll = loglik(x, obs_lik)

# Per-observation log-likelihoods (vector)
per_obs_ll = pointwise_loglik(x, obs_lik)

# These are equivalent
@assert sum(per_obs_ll) ≈ total_ll

# In-place version for performance-critical code
result = zeros(4)
pointwise_loglik!(result, x, obs_lik)
```

#### Conditional Independence

Pointwise log-likelihoods are only well-defined when observations are **conditionally independent** given the latent field. All current observation models in the package assume this property.

You can check the independence structure using the trait system:

```julia
obs_independence = observation_independence(obs_lik)

# All current models return ConditionallyIndependent()
obs_independence isa ConditionallyIndependent  # true
```

Future extensions might add models with `ConditionallyDependent` observations (e.g., multivariate normal with correlated noise), which would not support `pointwise_loglik`.

## Exponential Family Models

The most common observation models are exponential family distributions connected to the latent field through link functions.

### Basic Usage

```julia
using GaussianMarkovRandomFields
using Distributions

# Poisson model for count data (canonical LogLink)
poisson_model = ExponentialFamily(Poisson)
x = [1.0, 2.0]  # Latent field (log-intensity due to LogLink)
y = [2, 7]      # Count observations
obs_lik = poisson_model(y)
ll = loglik(x, obs_lik)

# Normal model for continuous data (canonical IdentityLink)
normal_model = ExponentialFamily(Normal)
x = [1.5, 2.3]  # Latent field (direct mean due to IdentityLink)
y = [1.2, 2.8]  # Continuous observations
obs_lik = normal_model(y; σ=0.5)  # Normal requires σ hyperparameter
ll = loglik(x, obs_lik)

# Bernoulli model for binary data (canonical LogitLink)
bernoulli_model = ExponentialFamily(Bernoulli)
x = [0.0, 1.5]  # Latent field (logit-probability due to LogitLink)
y = [0, 1]      # Binary observations
obs_lik = bernoulli_model(y)
ll = loglik(x, obs_lik)
```

### Supported Distributions and Links

| Distribution | Canonical Link | Alternative Links | Hyperparameters |
|-------------|----------------|-------------------|------------------|
| Normal      | IdentityLink   | LogLink          | σ (std. dev.)    |
| Poisson     | LogLink        | IdentityLink     | none             |
| Bernoulli   | LogitLink      | LogLink          | none             |
| Binomial    | LogitLink      | IdentityLink     | none*            |

*For Binomial, the number of trials is provided through the data structure `BinomialObservations`, not as a hyperparameter.

### Custom Link Functions

```julia
# Non-canonical link function
poisson_identity = ExponentialFamily(Poisson, IdentityLink())
# Note: Requires positive latent field values for valid Poisson intensities
```

## Custom Observation Models

For models not covered by exponential families, you can define custom log-likelihood functions using automatic differentiation.

### Basic AutoDiff Models

```julia
# Define custom log-likelihood function
function custom_loglik(x; y=[1.0, 2.0], σ=1.0)
    μ = sin.(x)  # Custom transformation
    return -0.5 * sum((y .- μ).^2) / σ^2 - length(y) * log(σ)
end

# Create observation model
obs_model = AutoDiffObservationModel(custom_loglik; n_latent=2, hyperparams=(:y, :σ))

# Materialize with data
obs_lik = obs_model(y=[1.2, 1.8], σ=0.5)

# Use normally - gradients and Hessians computed automatically!
x = [0.5, 1.0]
ll = loglik(x, obs_lik)
grad = loggrad(x, obs_lik)    # Automatic differentiation
hess = loghessian(x, obs_lik) # Potentially sparse!
```

### Pointwise Log-Likelihoods for AutoDiff Models

For model comparison metrics, you'll need to provide a pointwise log-likelihood function:

```julia
# Define both total and pointwise log-likelihood
function custom_loglik(x; y=[1.0, 2.0], σ=1.0)
    μ = sin.(x)
    return -0.5 * sum((y .- μ).^2) / σ^2 - length(y) * log(σ)
end

function custom_pointwise_loglik(x; y=[1.0, 2.0], σ=1.0)
    μ = sin.(x)
    return -0.5 .* (y .- μ).^2 / σ^2 .- log(σ)  # Per-observation
end

# Create model with pointwise support
obs_model = AutoDiffObservationModel(
    custom_loglik;
    n_latent=2,
    hyperparams=(:y, :σ),
    pointwise_loglik_func=custom_pointwise_loglik  # Enable pointwise!
)

obs_lik = obs_model(y=[1.2, 1.8], σ=0.5)
per_obs = pointwise_loglik(x, obs_lik)  # Now supported!
```

If you don't provide `pointwise_loglik_func`, attempting to call `pointwise_loglik` will error with a helpful message.

### Automatic Differentiation Requirements

AutoDiff observation models require an automatic differentiation backend. We support and recommend the following backends in order of preference:

1. **Enzyme.jl** (recommended for performance)
2. **Mooncake.jl** (good balance of performance and compatibility)
3. **Zygote.jl** (reliable fallback)
4. **ForwardDiff.jl** (for small problems)

```julia
# Load an AD backend (required for AutoDiffObservationModel)
using Enzyme  # Recommended

# Or use another supported backend:
# using Mooncake
# using Zygote
# using ForwardDiff

# Now you can use AutoDiff models
obs_model = AutoDiffObservationModel(my_loglik; n_latent=10)
obs_lik = obs_model(y=data)
grad = loggrad(x, obs_lik)  # Uses your loaded AD backend
```

### Sparse Hessian Computation

AutoDiff observation models can automatically detect and exploit sparsity in Hessian matrices using our package extensions. This requires loading both an AD backend and additional sparsity packages:

```julia
# Load AD backend + sparse AD packages
using Enzyme  # Or your preferred AD backend
using SparseConnectivityTracer, SparseMatrixColorings

# The package extension is automatically activated
obs_model = AutoDiffObservationModel(my_loglik; n_latent=100)
obs_lik = obs_model(y=data)

# Hessian computation now automatically:
# - Detects sparsity pattern using TracerSparsityDetector  
# - Uses greedy coloring for efficient computation
# - Returns sparse matrix when beneficial
hess = loghessian(x, obs_lik)  # May be sparse!
```

The sparse Hessian features provide dramatic performance improvements for large-scale problems with structured sparsity.

## Nonlinear Least Squares

Use when observations are Gaussian with mean given by an arbitrary (possibly nonlinear) function of the latent field: `y | x ~ Normal(f(x), σ)`.

Key properties
- Out-of-place `f`: define `f(x)::AbstractVector` with length equal to `length(y)`.
- Gauss–Newton: gradient and Hessian use the Gauss–Newton approximation (no exact Hessian term).
  - `∇ℓ(x) = J(x)' (w ⊙ r)`, where `r = y − f(x)`, `w = 1 ./ σ.^2`
  - `∇²ℓ(x) ≈ − J(x)' Diagonal(w) J(x)`
- `σ`: accepts a scalar or vector (heteroskedastic), both interpreted as standard deviations.
- Sparse autodiff: requires loading `SparseConnectivityTracer` and `SparseMatrixColorings` to activate the sparse Jacobian backend.

Example
```julia
using GaussianMarkovRandomFields
using SparseConnectivityTracer, SparseMatrixColorings  # activate sparse Jacobian backend

# Nonlinear mapping f: R^2 -> R^3
f(x) = [x[1] + 2x[2], sin(x[1]), x[2]^2]

# Observations and noise
y = [1.0, 0.5, 2.0]
σ = [0.3, 0.4, 0.5]  # vector sigma allowed

# Build model and materialize likelihood
model = NonlinearLeastSquaresModel(f, 2)
lik = model(y; σ=σ)

# Evaluate
x = [0.1, 0.2]
ll = loglik(x, lik)
g  = loggrad(x, lik)     # uses sparse DI.jacobian under the hood
H  = loghessian(x, lik)  # Gauss–Newton: -J' W J

# Conditional distribution p(y | x)
dist = conditional_distribution(model, x; σ=0.3)
```

API
```@docs
NonlinearLeastSquaresModel
NonlinearLeastSquaresLikelihood
```

## Advanced Features

### Linear Transformations and Design Matrices

For GLM-style modeling where observations are related to linear combinations of latent field components:

```julia
# Design matrix mapping latent field to linear predictors
# Rows = observations, Columns = latent components
A = [1.0  20.0  1.0  0.0;   # obs 1: intercept + temp + group1
     1.0  25.0  1.0  0.0;   # obs 2: intercept + temp + group1  
     1.0  30.0  0.0  1.0]   # obs 3: intercept + temp + group2

base_model = ExponentialFamily(Poisson)  # LogLink by default
obs_model = LinearlyTransformedObservationModel(base_model, A)

# Latent field now includes all components: [β₀, β₁, u₁, u₂]
x_full = [2.0, 0.1, -0.5, 0.3]  # intercept, slope, group effects
obs_lik = obs_model(y)
ll = loglik(x_full, obs_lik)  # Chain rule applied automatically
```

### Binomial Observations

For binomial data, use the `BinomialObservations` utility:

```julia
# Create binomial observations with successes and trials
y = BinomialObservations([3, 1, 4], [5, 8, 6])  # (successes, trials) pairs

# Use with Binomial model
binomial_model = ExponentialFamily(Binomial) 
obs_lik = binomial_model(y)

# Access components
successes(y)  # [3, 1, 4]
trials(y)     # [5, 8, 6]
y[1]          # (3, 5) - tuple access
```

### Composite Observations

For multiple observation types in a single model:

```julia
# Multiple observation vectors
count_data = [1, 3, 0, 2]
binary_data = [0, 1, 1, 0]
obs = CompositeObservations(count_data, binary_data)

# Corresponding models for each observation type
poisson_model = ExponentialFamily(Poisson)
bernoulli_model = ExponentialFamily(Bernoulli) 
composite_model = CompositeObservationModel(poisson_model, bernoulli_model)

obs_lik = composite_model(obs)
# Latent field x now corresponds to concatenated observations
ll = loglik(x, obs_lik)
```

## API Reference

### Core Types and Interface

```@docs
ObservationModel
ObservationLikelihood
hyperparameters(::ObservationModel)
latent_dimension
loglik
loggrad
loghessian
pointwise_loglik
pointwise_loglik!
```

### Observation Independence

```@docs
ObservationIndependence
ConditionallyIndependent
ConditionallyDependent
observation_independence
```

### Exponential Family Models

```@docs
ExponentialFamily
conditional_distribution
ExponentialFamilyLikelihood
NormalLikelihood
PoissonLikelihood
BernoulliLikelihood
BinomialLikelihood
```

### Link Functions

```@docs
LinkFunction
IdentityLink
LogLink
LogitLink
apply_link
apply_invlink
```

### Custom AutoDiff Models

```@docs
AutoDiffObservationModel
AutoDiffLikelihood
```

### FEM Helper Functions

```@docs
PointEvaluationObsModel
PointDerivativeObsModel
PointSecondDerivativeObsModel
```

### Advanced Features

```@docs
LinearlyTransformedObservationModel
LinearlyTransformedLikelihood
BinomialObservations
successes
trials
CompositeObservations
CompositeObservationModel
CompositeLikelihood
```
