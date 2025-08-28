
# Observation Models {#Observation-Models}

Observation models define the relationship between observations `y` and the latent GMRF field `x`, typically through likelihood functions. They enable Bayesian inference by connecting your data to the underlying Gaussian process through flexible probabilistic models.

GaussianMarkovRandomFields.jl implements observation models using a **factory pattern** that separates model configuration from materialized evaluation instances. This design provides major performance benefits in optimization loops and cleaner automatic differentiation boundaries.

## Core Concepts {#Core-Concepts}

### The Factory Pattern {#The-Factory-Pattern}

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

### Evaluation Interface {#Evaluation-Interface}

All materialized observation likelihoods support a common interface:
- `loglik(x, obs_lik)`: Evaluate log-likelihood 
  
- `loggrad(x, obs_lik)`: Compute gradient with respect to latent field
  
- `loghessian(x, obs_lik)`: Compute Hessian matrix
  

## Exponential Family Models {#Exponential-Family-Models}

The most common observation models are exponential family distributions connected to the latent field through link functions.

### Basic Usage {#Basic-Usage}

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


### Supported Distributions and Links {#Supported-Distributions-and-Links}

| Distribution | Canonical Link | Alternative Links | Hyperparameters |
| ------------:| --------------:| -----------------:| ---------------:|
|       Normal |   IdentityLink |           LogLink |   σ (std. dev.) |
|      Poisson |        LogLink |      IdentityLink |            none |
|    Bernoulli |      LogitLink |           LogLink |            none |
|     Binomial |      LogitLink |      IdentityLink |           none* |


*For Binomial, the number of trials is provided through the data structure `BinomialObservations`, not as a hyperparameter.

### Custom Link Functions {#Custom-Link-Functions}

```julia
# Non-canonical link function
poisson_identity = ExponentialFamily(Poisson, IdentityLink())
# Note: Requires positive latent field values for valid Poisson intensities
```


## Custom Observation Models {#Custom-Observation-Models}

For models not covered by exponential families, you can define custom log-likelihood functions using automatic differentiation.

### Basic AutoDiff Models {#Basic-AutoDiff-Models}

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


### Automatic Differentiation Requirements {#Automatic-Differentiation-Requirements}

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


### Sparse Hessian Computation {#Sparse-Hessian-Computation}

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

## Advanced Features {#Advanced-Features}

### Linear Transformations and Design Matrices {#Linear-Transformations-and-Design-Matrices}

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


### Binomial Observations {#Binomial-Observations}

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


### Composite Observations {#Composite-Observations}

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


## API Reference {#API-Reference}

### Core Types and Interface {#Core-Types-and-Interface}
<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.ObservationModel' href='#GaussianMarkovRandomFields.ObservationModel'><span class="jlbinding">GaussianMarkovRandomFields.ObservationModel</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
ObservationModel
```


Abstract base type for all observation models for GMRFs.

An observation model defines the relationship between observations `y` and the latent field `x`, typically through a likelihood function. ObservationModel types serve as factories for creating ObservationLikelihood instances via callable syntax.

**Usage Pattern**

```julia
# Step 1: Create observation model (factory)
obs_model = ExponentialFamily(Normal)

# Step 2: Materialize with data and hyperparameters
obs_lik = obs_model(y; σ=1.2)  # Creates ObservationLikelihood

# Step 3: Use materialized likelihood in hot loops
ll = loglik(x, obs_lik)  # Fast x-only evaluation
```


See also: [`ObservationLikelihood`](/reference/observation_models#GaussianMarkovRandomFields.ObservationLikelihood), [`ExponentialFamily`](/reference/observation_models#GaussianMarkovRandomFields.ExponentialFamily)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/observation_models/observation_model.jl#L5-L27" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.ObservationLikelihood' href='#GaussianMarkovRandomFields.ObservationLikelihood'><span class="jlbinding">GaussianMarkovRandomFields.ObservationLikelihood</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
ObservationLikelihood
```


Abstract base type for materialized observation likelihoods.

Observation likelihoods are created by materializing an observation model with specific hyperparameters θ and observed data y. They provide efficient evaluation methods that  only depend on the latent field x, eliminating the need to repeatedly pass θ and y.

This design provides major performance benefits in optimization loops and cleaner  automatic differentiation boundaries.

**Usage Pattern**

```julia
# Step 1: Configure observation model (factory)
obs_model = ExponentialFamily(Normal)

# Step 2: Materialize with data and hyperparameters  
obs_lik = obs_model(y; σ=1.2)

# Step 3: Fast evaluation in hot loops
ll = loglik(x, obs_lik)      # Only x argument needed!
grad = loggrad(x, obs_lik)   # Fast x-only evaluation
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/observation_models/observation_likelihood.jl#L7-L32" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.hyperparameters-Tuple{ObservationModel}' href='#GaussianMarkovRandomFields.hyperparameters-Tuple{ObservationModel}'><span class="jlbinding">GaussianMarkovRandomFields.hyperparameters</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
hyperparameters(obs_model::ObservationModel) -> Tuple{Vararg{Symbol}}
```


Return a tuple of required hyperparameter names for this observation model.

This method defines which hyperparameters the observation model expects to receive when materializing an ObservationLikelihood instance.

**Arguments**
- `obs_model`: An observation model implementing the `ObservationModel` interface
  

**Returns**
- `Tuple{Vararg{Symbol}}`: Tuple of parameter names (e.g., `(:σ,)` or `(:α, :β)`)
  

**Example**

```julia
hyperparameters(ExponentialFamily(Normal)) == (:σ,)
hyperparameters(ExponentialFamily(Bernoulli)) == ()
```


**Implementation**

All observation models should implement this method. The default returns an empty tuple.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/observation_models/observation_model.jl#L51-L73" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.latent_dimension' href='#GaussianMarkovRandomFields.latent_dimension'><span class="jlbinding">GaussianMarkovRandomFields.latent_dimension</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
latent_dimension(obs_model::ObservationModel, y::AbstractVector) -> Union{Int, Nothing}
```


Return the latent field dimension for this observation model given observations y.

For most observation models, this will be `length(y)` (1:1 mapping). For transformed observation models like `LinearlyTransformedObservationModel`, this will be the dimension of the design matrix.

Returns `nothing` if the latent dimension cannot be determined automatically.

**Arguments**
- `obs_model`: An observation model implementing the `ObservationModel` interface
  
- `y`: Vector of observations
  

**Returns**
- `Int`: The latent field dimension, or `nothing` if unknown
  

**Example**

```julia
latent_dimension(ExponentialFamily(Normal), y) == length(y)
latent_dimension(LinearlyTransformedObservationModel(base, A), y) == size(A, 2)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/observation_models/observation_model.jl#L76-L99" target="_blank" rel="noreferrer">source</a></Badge>



```julia
latent_dimension(ef::ExponentialFamily, y::AbstractVector) -> Int
```


Return the latent field dimension for exponential family models.

For ExponentialFamily models, there is a direct 1:1 mapping between observations and latent field components, so the latent dimension equals the observation dimension.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/observation_models/exponential_family/exponential_family.jl#L204-L211" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.loglik' href='#GaussianMarkovRandomFields.loglik'><span class="jlbinding">GaussianMarkovRandomFields.loglik</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
loglik(x, lik::ExponentialFamilyLikelihood) -> Float64
```


Generic loglik implementation for all exponential family likelihoods using product_distribution.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/observation_models/exponential_family/canonical_implementations.jl#L12-L16" target="_blank" rel="noreferrer">source</a></Badge>



```julia
loglik(x, lik::NormalLikelihood) -> Float64
```


Specialized fast implementation for Normal likelihood that avoids product_distribution overhead.

Computes: ∑ᵢ logpdf(Normal(μᵢ, σ), yᵢ) = -n/2 * log(2π) - n * log(σ) - 1/(2σ²) * ∑ᵢ(yᵢ - μᵢ)²


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/observation_models/exponential_family/canonical_implementations.jl#L26-L32" target="_blank" rel="noreferrer">source</a></Badge>



```julia
loglik(x, composite_lik::CompositeLikelihood) -> Float64
```


Compute the log-likelihood of a composite likelihood by summing component contributions.

Each component likelihood receives the full latent field `x` and contributes to the total log-likelihood. This handles cases where components may have overlapping dependencies on the latent field.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/observation_models/composite/composite_evaluation.jl#L8-L16" target="_blank" rel="noreferrer">source</a></Badge>



```julia
loglik(x, obs_lik::AutoDiffLikelihood) -> Real
```


Evaluate the log-likelihood function at latent field `x`.

Calls the stored log-likelihood function, which typically includes all necessary hyperparameters and data as a closure.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/observation_models/autodiff_likelihood.jl#L224-L231" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.loggrad' href='#GaussianMarkovRandomFields.loggrad'><span class="jlbinding">GaussianMarkovRandomFields.loggrad</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
loggrad(x, obs_lik::ObservationLikelihood) -> Vector{Float64}
```


Automatic differentiation fallback for ObservationLikelihood gradient computation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/observation_models/observation_likelihood.jl#L45-L49" target="_blank" rel="noreferrer">source</a></Badge>



```julia
loggrad(x, composite_lik::CompositeLikelihood) -> Vector{Float64}
```


Compute the gradient of the log-likelihood by summing component gradients.

Each component contributes its gradient with respect to the full latent field `x`. For overlapping dependencies, gradients are automatically summed at each latent field element.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/observation_models/composite/composite_evaluation.jl#L21-L28" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.loghessian' href='#GaussianMarkovRandomFields.loghessian'><span class="jlbinding">GaussianMarkovRandomFields.loghessian</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
loghessian(x, obs_lik::ObservationLikelihood) -> AbstractMatrix{Float64}
```


Automatic differentiation fallback for ObservationLikelihood Hessian computation.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/observation_models/observation_likelihood.jl#L68-L72" target="_blank" rel="noreferrer">source</a></Badge>



```julia
loghessian(x, composite_lik::CompositeLikelihood) -> AbstractMatrix{Float64}
```


Compute the Hessian of the log-likelihood by summing component Hessians.

Each component contributes its Hessian with respect to the full latent field `x`. For overlapping dependencies, Hessians are automatically summed element-wise.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/observation_models/composite/composite_evaluation.jl#L41-L48" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Exponential Family Models {#Exponential-Family-Models-2}
<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.ExponentialFamily' href='#GaussianMarkovRandomFields.ExponentialFamily'><span class="jlbinding">GaussianMarkovRandomFields.ExponentialFamily</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
ExponentialFamily{F<:Distribution, L<:LinkFunction} <: ObservationModel
```


Observation model for exponential family distributions with link functions.

This struct represents observation models where the observations come from an exponential  family distribution (Normal, Poisson, Bernoulli, Binomial) and the mean parameter is  related to the latent field through a link function.

**Mathematical Model**

For observations yᵢ and latent field values xᵢ:
- Linear predictor: ηᵢ = xᵢ
  
- Mean parameter: μᵢ = g⁻¹(ηᵢ) where g is the link function
  
- Observations: yᵢ ~ F(μᵢ, θ) where F is the distribution family
  

**Fields**
- `family::Type{F}`: The distribution family (e.g., `Poisson`, `Bernoulli`)
  
- `link::L`: The link function connecting mean parameters to linear predictors
  

**Type Parameters**
- `F`: A subtype of `Distribution` from Distributions.jl
  
- `L`: A subtype of `LinkFunction`
  

**Constructors**

```julia
# Use canonical link (recommended)
ExponentialFamily(Poisson)        # Uses LogLink()
ExponentialFamily(Bernoulli)      # Uses LogitLink()
ExponentialFamily(Normal)         # Uses IdentityLink()

# Specify custom link function
ExponentialFamily(Poisson, IdentityLink())  # Non-canonical
```


**Supported Combinations**
- `Normal` with `IdentityLink` (canonical) or `LogLink`
  
- `Poisson` with `LogLink` (canonical) or `IdentityLink`  
  
- `Bernoulli` with `LogitLink` (canonical) or `LogLink`
  
- `Binomial` with `LogitLink` (canonical) or `IdentityLink`
  

**Hyperparameters (θ)**

Different families require different hyperparameters:
- `Normal`: `θ = [σ]` (standard deviation)
  
- `Poisson`: `θ = []` (no hyperparameters)
  
- `Bernoulli`: `θ = []` (no hyperparameters)
  
- `Binomial`: `θ = [n]` (number of trials)
  

**Examples**

```julia
# Poisson model for count data
model = ExponentialFamily(Poisson)
x = [1.0, 2.0]        # Latent field (log scale due to LogLink)
θ = Float64[]         # No hyperparameters  
y = [2, 7]           # Count observations

ll = loglik(x, model, θ, y)
dist = data_distribution(model, x, θ)  # Returns Product distribution

# Bernoulli model for binary data
model = ExponentialFamily(Bernoulli)
x = [0.0, 1.0]       # Latent field (logit scale due to LogitLink)
y = [0, 1]           # Binary observations
```


**Performance Notes**

Canonical link functions have optimized implementations that avoid redundant computations. Non-canonical links use general chain rule formulations which may be slower.

See also: [`LinkFunction`](/reference/observation_models#GaussianMarkovRandomFields.LinkFunction), [`loglik`](/reference/observation_models#GaussianMarkovRandomFields.loglik), [`data_distribution`](/reference/observation_models#GaussianMarkovRandomFields.data_distribution)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/observation_models/exponential_family/exponential_family.jl#L7-L76" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.data_distribution' href='#GaussianMarkovRandomFields.data_distribution'><span class="jlbinding">GaussianMarkovRandomFields.data_distribution</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
data_distribution(obs_model::ExponentialFamily, x, θ_named) -> Distribution
```


Construct the data-generating distribution p(y | x, θ).

This function returns a Distribution object that represents the probability  distribution over observations y given latent field values x and hyperparameters θ. It is used for sampling new observations.

**Arguments**
- `obs_model`: An ExponentialFamily observation model
  
- `x`: Latent field values (vector)  
  
- `θ_named`: Hyperparameters as a NamedTuple
  

**Returns**

Distribution object that can be used with `rand()` to generate observations

**Example**

```julia
model = ExponentialFamily(Poisson)
x = [1.0, 2.0]
θ_named = NamedTuple()
dist = data_distribution(model, x, θ_named)
y = rand(dist)  # Sample observations
```


Note: For likelihood evaluation L(x|y,θ), use the materialized API:

```julia
obs_lik = obs_model(y; θ_named...)
ll = loglik(x, obs_lik)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/observation_models/exponential_family/exponential_family.jl#L121-L152" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.ExponentialFamilyLikelihood' href='#GaussianMarkovRandomFields.ExponentialFamilyLikelihood'><span class="jlbinding">GaussianMarkovRandomFields.ExponentialFamilyLikelihood</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
ExponentialFamilyLikelihood{L, I} <: ObservationLikelihood
```


Abstract type for exponential family observation likelihoods.

This intermediate type allows for generic implementations that work across all  exponential family distributions while still allowing specialized methods for  specific combinations.

**Type Parameters**
- `L`: Link function type
  
- `I`: Index type (Nothing for non-indexed, UnitRange or Vector for indexed)
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/observation_models/exponential_family/observation_likelihoods.jl#L3-L15" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.NormalLikelihood' href='#GaussianMarkovRandomFields.NormalLikelihood'><span class="jlbinding">GaussianMarkovRandomFields.NormalLikelihood</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
NormalLikelihood{L<:LinkFunction} <: ObservationLikelihood
```


Materialized Normal observation likelihood with precomputed hyperparameters.

**Fields**
- `link::L`: Link function connecting latent field to mean parameter
  
- `y::Vector{Float64}`: Observed data  
  
- `σ::Float64`: Standard deviation hyperparameter
  
- `inv_σ²::Float64`: Precomputed 1/σ² for performance
  
- `log_σ::Float64`: Precomputed log(σ) for log-likelihood computation
  

**Example**

```julia
obs_model = ExponentialFamily(Normal)
obs_lik = obs_model([1.0, 2.0, 1.5]; σ=0.5)  # NormalLikelihood{IdentityLink}
ll = loglik([0.9, 2.1, 1.4], obs_lik)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/observation_models/exponential_family/observation_likelihoods.jl#L18-L36" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.PoissonLikelihood' href='#GaussianMarkovRandomFields.PoissonLikelihood'><span class="jlbinding">GaussianMarkovRandomFields.PoissonLikelihood</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
PoissonLikelihood{L<:LinkFunction} <: ObservationLikelihood
```


Materialized Poisson observation likelihood.

**Fields**
- `link::L`: Link function connecting latent field to rate parameter
  
- `y::Vector{Int}`: Count observations
  

**Example**

```julia
obs_model = ExponentialFamily(Poisson)  # Uses LogLink by default
obs_lik = obs_model([1, 3, 0, 2])      # PoissonLikelihood{LogLink}
ll = loglik([0.0, 1.1, -2.0, 0.7], obs_lik)  # x values on log scale
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/observation_models/exponential_family/observation_likelihoods.jl#L46-L61" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.BernoulliLikelihood' href='#GaussianMarkovRandomFields.BernoulliLikelihood'><span class="jlbinding">GaussianMarkovRandomFields.BernoulliLikelihood</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
BernoulliLikelihood{L<:LinkFunction} <: ObservationLikelihood
```


Materialized Bernoulli observation likelihood for binary data.

**Fields**
- `link::L`: Link function connecting latent field to probability parameter  
  
- `y::Vector{Int}`: Binary observations (0 or 1)
  

**Example**

```julia
obs_model = ExponentialFamily(Bernoulli)  # Uses LogitLink by default
obs_lik = obs_model([1, 0, 1, 0])        # BernoulliLikelihood{LogitLink}
ll = loglik([0.5, -0.2, 1.1, -0.8], obs_lik)  # x values on logit scale
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/observation_models/exponential_family/observation_likelihoods.jl#L68-L83" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.BinomialLikelihood' href='#GaussianMarkovRandomFields.BinomialLikelihood'><span class="jlbinding">GaussianMarkovRandomFields.BinomialLikelihood</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
BinomialLikelihood{L<:LinkFunction} <: ObservationLikelihood
```


Materialized Binomial observation likelihood.

**Fields**
- `link::L`: Link function connecting latent field to probability parameter
  
- `y::Vector{Int}`: Number of successes for each trial
  
- `n::Vector{Int}`: Number of trials per observation (can vary across observations)
  

**Example**

```julia
obs_model = ExponentialFamily(Binomial)  # Uses LogitLink by default
obs_lik = obs_model([3, 1, 4]; trials=[5, 8, 6])  # BinomialLikelihood{LogitLink}
ll = loglik([0.2, -1.0, 0.8], obs_lik)  # x values on logit scale
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/observation_models/exponential_family/observation_likelihoods.jl#L90-L106" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Link Functions {#Link-Functions}
<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.LinkFunction' href='#GaussianMarkovRandomFields.LinkFunction'><span class="jlbinding">GaussianMarkovRandomFields.LinkFunction</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
LinkFunction
```


Abstract base type for link functions used in exponential family models.

A link function g(μ) connects the mean parameter μ of a distribution to the linear  predictor η through the relationship g(μ) = η, or equivalently μ = g⁻¹(η).

**Implemented Link Functions**
- [`IdentityLink`](/reference/observation_models#GaussianMarkovRandomFields.IdentityLink): g(μ) = μ (for Normal distributions)
  
- [`LogLink`](/reference/observation_models#GaussianMarkovRandomFields.LogLink): g(μ) = log(μ) (for Poisson distributions)  
  
- [`LogitLink`](/reference/observation_models#GaussianMarkovRandomFields.LogitLink): g(μ) = logit(μ) (for Bernoulli/Binomial distributions)
  

**Interface**

Concrete link functions must implement:
- `apply_link(link, μ)`: Apply the link function g(μ)
  
- `apply_invlink(link, η)`: Apply the inverse link function g⁻¹(η)
  

For performance in GMRF computations, they should also implement:
- `derivative_invlink(link, η)`: First derivative of g⁻¹(η)
  
- `second_derivative_invlink(link, η)`: Second derivative of g⁻¹(η)
  

See also: [`ExponentialFamily`](/reference/observation_models#GaussianMarkovRandomFields.ExponentialFamily), [`apply_link`](/reference/observation_models#GaussianMarkovRandomFields.apply_link), [`apply_invlink`](/reference/observation_models#GaussianMarkovRandomFields.apply_invlink)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/observation_models/exponential_family/link_functions.jl#L6-L29" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.IdentityLink' href='#GaussianMarkovRandomFields.IdentityLink'><span class="jlbinding">GaussianMarkovRandomFields.IdentityLink</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
IdentityLink <: LinkFunction
```


Identity link function: g(μ) = μ.

This is the canonical link for Normal distributions. The mean parameter μ is  directly equal to the linear predictor η.

**Mathematical Definition**
- Link: g(μ) = μ
  
- Inverse link: g⁻¹(η) = η  
  
- First derivative: d/dη g⁻¹(η) = 1
  
- Second derivative: d²/dη² g⁻¹(η) = 0
  

**Example**

```julia
link = IdentityLink()
μ = apply_invlink(link, 1.5)  # μ = 1.5
η = apply_link(link, μ)       # η = 1.5
```


See also: [`LogLink`](/reference/observation_models#GaussianMarkovRandomFields.LogLink), [`LogitLink`](/reference/observation_models#GaussianMarkovRandomFields.LogitLink)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/observation_models/exponential_family/link_functions.jl#L32-L54" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.LogLink' href='#GaussianMarkovRandomFields.LogLink'><span class="jlbinding">GaussianMarkovRandomFields.LogLink</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
LogLink <: LinkFunction
```


Logarithmic link function: g(μ) = log(μ).

This is the canonical link for Poisson and Gamma distributions. It ensures the  mean parameter μ remains positive by mapping the real-valued linear predictor η  to μ = exp(η).

**Mathematical Definition**
- Link: g(μ) = log(μ) 
  
- Inverse link: g⁻¹(η) = exp(η)
  
- First derivative: d/dη g⁻¹(η) = exp(η)
  
- Second derivative: d²/dη² g⁻¹(η) = exp(η)
  

**Example**

```julia
link = LogLink()
μ = apply_invlink(link, 1.0)  # μ = exp(1.0) ≈ 2.718
η = apply_link(link, μ)       # η = log(μ) = 1.0
```


See also: [`IdentityLink`](/reference/observation_models#GaussianMarkovRandomFields.IdentityLink), [`LogitLink`](/reference/observation_models#GaussianMarkovRandomFields.LogitLink)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/observation_models/exponential_family/link_functions.jl#L57-L80" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.LogitLink' href='#GaussianMarkovRandomFields.LogitLink'><span class="jlbinding">GaussianMarkovRandomFields.LogitLink</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
LogitLink <: LinkFunction
```


Logit link function: g(μ) = logit(μ) = log(μ/(1-μ)).

This is the canonical link for Bernoulli and Binomial distributions. It maps  probabilities μ ∈ (0,1) to the real line via the logistic transformation,  ensuring μ = logistic(η) = 1/(1+exp(-η)) remains a valid probability.

**Mathematical Definition**
- Link: g(μ) = logit(μ) = log(μ/(1-μ))
  
- Inverse link: g⁻¹(η) = logistic(η) = 1/(1+exp(-η))
  
- First derivative: d/dη g⁻¹(η) = μ(1-μ) where μ = logistic(η)
  
- Second derivative: d²/dη² g⁻¹(η) = μ(1-μ)(1-2μ)
  

**Example**

```julia
link = LogitLink()
μ = apply_invlink(link, 0.0)  # μ = logistic(0.0) = 0.5
η = apply_link(link, μ)       # η = logit(0.5) = 0.0
```


See also: [`IdentityLink`](/reference/observation_models#GaussianMarkovRandomFields.IdentityLink), [`LogLink`](/reference/observation_models#GaussianMarkovRandomFields.LogLink)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/observation_models/exponential_family/link_functions.jl#L83-L106" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.apply_link' href='#GaussianMarkovRandomFields.apply_link'><span class="jlbinding">GaussianMarkovRandomFields.apply_link</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
apply_link(link::LinkFunction, μ) -> Real
```


Apply the link function g(μ) to transform mean parameters to linear predictor scale.

This function computes η = g(μ), where g is the link function. This transformation is typically used to ensure the mean parameter satisfies appropriate constraints (e.g., positivity for Poisson, probability bounds for Bernoulli).

**Arguments**
- `link`: A link function (IdentityLink, LogLink, or LogitLink)
  
- `μ`: Mean parameter value(s) in the natural parameter space
  

**Returns**

The transformed value(s) η on the linear predictor scale

**Examples**

```julia
apply_link(LogLink(), 2.718)      # ≈ 1.0
apply_link(LogitLink(), 0.5)      # = 0.0  
apply_link(IdentityLink(), 1.5)   # = 1.5
```


See also: [`apply_invlink`](/reference/observation_models#GaussianMarkovRandomFields.apply_invlink), [`LinkFunction`](/reference/observation_models#GaussianMarkovRandomFields.LinkFunction)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/observation_models/exponential_family/link_functions.jl#L109-L133" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.apply_invlink' href='#GaussianMarkovRandomFields.apply_invlink'><span class="jlbinding">GaussianMarkovRandomFields.apply_invlink</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
apply_invlink(link::LinkFunction, η) -> Real
```


Apply the inverse link function g⁻¹(η) to transform linear predictor to mean parameters.

This function computes μ = g⁻¹(η), where g⁻¹ is the inverse link function. This is the primary transformation used in GMRF models to convert the latent field values to the natural parameter space of the observation distribution.

**Arguments**
- `link`: A link function (IdentityLink, LogLink, or LogitLink)
  
- `η`: Linear predictor value(s)
  

**Returns**

The transformed value(s) μ in the natural parameter space

**Examples**

```julia
apply_invlink(LogLink(), 1.0)      # ≈ 2.718 (= exp(1))
apply_invlink(LogitLink(), 0.0)    # = 0.5   (= logistic(0))
apply_invlink(IdentityLink(), 1.5) # = 1.5
```


See also: [`apply_link`](/reference/observation_models#GaussianMarkovRandomFields.apply_link), [`LinkFunction`](/reference/observation_models#GaussianMarkovRandomFields.LinkFunction)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/observation_models/exponential_family/link_functions.jl#L138-L162" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Custom AutoDiff Models {#Custom-AutoDiff-Models}
<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.AutoDiffObservationModel' href='#GaussianMarkovRandomFields.AutoDiffObservationModel'><span class="jlbinding">GaussianMarkovRandomFields.AutoDiffObservationModel</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
AutoDiffObservationModel{F, B, SB, H} <: ObservationModel
```


Observation model that uses automatic differentiation for a user-provided log-likelihood function.

This serves as a factory for creating AutoDiffLikelihood instances. The user provides a  log-likelihood function that can accept hyperparameters, and when materialized, creates a closure with the hyperparameters baked in.

**Type Parameters**
- `F`: Type of the log-likelihood function  
  
- `B`: Type of the AD backend for gradients
  
- `SB`: Type of the AD backend for Hessians
  
- `H`: Type of the hyperparameters tuple
  

**Fields**
- `loglik_func::F`: User-provided log-likelihood function with signature `(x; kwargs...) -> Real`
  
- `n_latent::Int`: Number of latent field components
  
- `grad_backend::B`: AD backend for gradient computation
  
- `hess_backend::SB`: AD backend for Hessian computation
  
- `hyperparams::H`: Tuple of hyperparameter names that this model expects
  

**Usage**

```julia
# Define your log-likelihood function with hyperparameters
function my_loglik(x; σ=1.0, y=[1.0, 2.0])
    μ = x  # or some transformation of x
    return -0.5 * sum((y .- μ).^2) / σ^2 - length(y) * log(σ)
end

# Create observation model specifying expected hyperparameters
obs_model = AutoDiffObservationModel(my_loglik; n_latent=2, hyperparams=(:σ, :y))

# Materialize with specific hyperparameters
obs_lik = obs_model(σ=0.5, y=[1.2, 1.8])  # Creates AutoDiffLikelihood

# Use normally
ll = loglik(x, obs_lik)
grad = loggrad(x, obs_lik)
hess = loghessian(x, obs_lik)
```


See also: [`AutoDiffLikelihood`](/reference/observation_models#GaussianMarkovRandomFields.AutoDiffLikelihood), [`ObservationModel`](/reference/observation_models#GaussianMarkovRandomFields.ObservationModel)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/observation_models/autodiff_likelihood.jl#L66-L109" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.AutoDiffLikelihood' href='#GaussianMarkovRandomFields.AutoDiffLikelihood'><span class="jlbinding">GaussianMarkovRandomFields.AutoDiffLikelihood</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
AutoDiffLikelihood{F, B, SB, GP, HP} <: ObservationLikelihood
```


Automatic differentiation-based observation likelihood that wraps a user-provided log-likelihood function.

This is a materialized likelihood created from an AutoDiffObservationModel. The log-likelihood function is typically a closure that already includes hyperparameters and data.

**Type Parameters**
- `F`: Type of the log-likelihood function (usually a closure)
  
- `B`: Type of the AD backend for gradients
  
- `SB`: Type of the AD backend for Hessians
  
- `GP`: Type of the gradient preparation object
  
- `HP`: Type of the Hessian preparation object
  

**Fields**
- `loglik_func::F`: Log-likelihood function with signature `(x) -> Real`
  
- `grad_backend::B`: AD backend for gradient computation
  
- `hess_backend::SB`: AD backend for Hessian computation
  
- `grad_prep::GP`: Preparation object for gradient computation
  
- `hess_prep::HP`: Preparation object for Hessian computation
  

**Usage**

Typically created via AutoDiffObservationModel factory:

```julia
# Define your log-likelihood function with hyperparameters
function my_loglik(x; σ=1.0, y=[1.0, 2.0])
    μ = x  # or some transformation of x
    return -0.5 * sum((y .- μ).^2) / σ^2 - length(y) * log(σ)
end

# Create observation model
obs_model = AutoDiffObservationModel(my_loglik, n_latent=2)

# Materialize with data and hyperparameters
obs_lik = obs_model(σ=0.5, y=[1.2, 1.8])  # Creates AutoDiffLikelihood

# Use in the standard way
x = [1.1, 1.9]
ll = loglik(x, obs_lik)
grad = loggrad(x, obs_lik)  # Uses prepared AD with optimal backends
hess = loghessian(x, obs_lik)  # Automatically sparse when available!
```


**Sparse Hessian Features**

The Hessian computation automatically:
- Detects sparsity pattern using TracerSparsityDetector
  
- Uses greedy coloring for efficient computation
  
- Returns a sparse matrix when beneficial
  
- Falls back to dense computation for small problems
  

See also: [`loglik`](/reference/observation_models#GaussianMarkovRandomFields.loglik), [`loggrad`](/reference/observation_models#GaussianMarkovRandomFields.loggrad), [`loghessian`](/reference/observation_models#GaussianMarkovRandomFields.loghessian)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/observation_models/autodiff_likelihood.jl#L5-L57" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Advanced Features {#Advanced-Features-2}
<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.LinearlyTransformedObservationModel' href='#GaussianMarkovRandomFields.LinearlyTransformedObservationModel'><span class="jlbinding">GaussianMarkovRandomFields.LinearlyTransformedObservationModel</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
LinearlyTransformedObservationModel{M, A} <: ObservationModel
```


Observation model that applies a linear transformation to the latent field before  passing to a base observation model. This enables GLM-style modeling with design  matrices while maintaining full compatibility with existing observation models.

**Mathematical Foundation**

The wrapper transforms the full latent field x_full to linear predictors η via a  design matrix A:
- η = A * x_full  
  
- Base model operates on η as usual: p(y | η, θ)
  
- Chain rule applied for gradients/Hessians: 
  - ∇_{x_full} ℓ = A^T ∇_η ℓ
    
  - ∇²_{x_full} ℓ = A^T ∇²_η ℓ A
    
  

**Type Parameters**
- `M <: ObservationModel`: Type of the base observation model
  
- `A`: Type of the design matrix (typically AbstractMatrix)
  

**Fields**
- `base_model::M`: The underlying observation model that operates on linear predictors
  
- `design_matrix::A`: Matrix mapping full latent field to observation-specific linear predictors
  

**Usage Pattern**

```julia
# Step 1: Create base observation model
base_model = ExponentialFamily(Poisson)  # LogLink by default

# Step 2: Create design matrix (maps latent field to linear predictors)
# For: y ~ intercept + temperature + group_effects
A = [1.0  20.0  1.0  0.0  0.0;   # obs 1: intercept + temp + group1
     1.0  25.0  1.0  0.0  0.0;   # obs 2: intercept + temp + group1  
     1.0  30.0  0.0  1.0  0.0;   # obs 3: intercept + temp + group2
     1.0  15.0  0.0  0.0  1.0]   # obs 4: intercept + temp + group3

# Step 3: Create wrapped model
obs_model = LinearlyTransformedObservationModel(base_model, A)

# Step 4: Use in GMRF model - latent field now includes all components
# x_full = [β₀, β₁, u₁, u₂, u₃]  # intercept, slope, group effects

# Step 5: Materialize with data and hyperparameters
obs_lik = obs_model(y; σ=1.2)  # Creates LinearlyTransformedLikelihood

# Step 6: Fast evaluation in optimization loops
ll = loglik(x_full, obs_lik)
```


**Hyperparameters**

All hyperparameters come from the base observation model. The design matrix  introduces no new hyperparameters - it&#39;s a fixed linear transformation.

See also: [`LinearlyTransformedLikelihood`](/reference/observation_models#GaussianMarkovRandomFields.LinearlyTransformedLikelihood), [`ExponentialFamily`](/reference/observation_models#GaussianMarkovRandomFields.ExponentialFamily), [`ObservationModel`](/reference/observation_models#GaussianMarkovRandomFields.ObservationModel)


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/observation_models/linearly_transformed.jl#L6-L60" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.LinearlyTransformedLikelihood' href='#GaussianMarkovRandomFields.LinearlyTransformedLikelihood'><span class="jlbinding">GaussianMarkovRandomFields.LinearlyTransformedLikelihood</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
LinearlyTransformedLikelihood{L, A} <: ObservationLikelihood
```


Materialized likelihood for LinearlyTransformedObservationModel with precomputed  base likelihood and design matrix.

This is created by calling a LinearlyTransformedObservationModel instance with  data and hyperparameters, following the factory pattern used throughout the package.

**Type Parameters**
- `L <: ObservationLikelihood`: Type of the materialized base likelihood
  
- `A`: Type of the design matrix
  

**Fields**
- `base_likelihood::L`: Materialized base observation likelihood (contains y and θ)
  
- `design_matrix::A`: Design matrix mapping full latent field to linear predictors
  

**Usage**

This type is typically created automatically:

```julia
ltom = LinearlyTransformedObservationModel(base_model, design_matrix)
ltlik = ltom(y; σ=1.2)  # Creates LinearlyTransformedLikelihood
ll = loglik(x_full, ltlik)  # Fast evaluation
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/observation_models/linearly_transformed.jl#L81-L105" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.BinomialObservations' href='#GaussianMarkovRandomFields.BinomialObservations'><span class="jlbinding">GaussianMarkovRandomFields.BinomialObservations</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
BinomialObservations <: AbstractVector{Tuple{Int, Int}}
```


Combined observation type for binomial data containing both successes and trials.

This type packages binomial observation data (number of successes and trials) into a single vector-like object where each element is a (successes, trials) tuple.

**Fields**
- `successes::Vector{Int}`: Number of successes for each observation
  
- `trials::Vector{Int}`: Number of trials for each observation
  

**Example**

```julia
# Create binomial observations
y = BinomialObservations([3, 1, 4], [5, 8, 6])

# Access as tuples
y[1]  # (3, 5)
y[2]  # (1, 8)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/observation_models/binomial_observations.jl#L3-L24" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.successes' href='#GaussianMarkovRandomFields.successes'><span class="jlbinding">GaussianMarkovRandomFields.successes</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
successes(y::BinomialObservations) -> Vector{Int}
```


Extract the successes vector from binomial observations.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/observation_models/binomial_observations.jl#L61-L65" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.trials' href='#GaussianMarkovRandomFields.trials'><span class="jlbinding">GaussianMarkovRandomFields.trials</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
trials(y::BinomialObservations) -> Vector{Int}
```


Extract the trials vector from binomial observations.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/observation_models/binomial_observations.jl#L68-L72" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.CompositeObservations' href='#GaussianMarkovRandomFields.CompositeObservations'><span class="jlbinding">GaussianMarkovRandomFields.CompositeObservations</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
CompositeObservations{T<:Tuple} <: AbstractVector{Float64}
```


A composite observation vector that stores observation data as a tuple of component vectors.

This type implements the `AbstractVector` interface and allows combining different observation datasets while maintaining their structure. The composite vector presents a unified view where indexing delegates to the appropriate component vector.

**Fields**
- `components::T`: Tuple of observation vectors, one per likelihood component
  

**Example**

```julia
y1 = [1.0, 2.0, 3.0]  # Gaussian observations
y2 = [4.0, 5.0]       # More Gaussian observations
y_composite = CompositeObservations((y1, y2))

length(y_composite)    # 5
y_composite[1]         # 1.0
y_composite[4]         # 4.0
collect(y_composite)   # [1.0, 2.0, 3.0, 4.0, 5.0]
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/observation_models/composite/composite_observations.jl#L1-L24" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.CompositeObservationModel' href='#GaussianMarkovRandomFields.CompositeObservationModel'><span class="jlbinding">GaussianMarkovRandomFields.CompositeObservationModel</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
CompositeObservationModel{T<:Tuple} <: ObservationModel
```


An observation model that combines multiple component observation models.

This type follows the factory pattern - it stores component observation models and  creates `CompositeLikelihood` instances when called with observation data and hyperparameters.

**Fields**
- `components::T`: Tuple of component observation models for type stability
  

**Example**

```julia
gaussian_model = ExponentialFamily(Normal)
poisson_model = ExponentialFamily(Poisson)
composite_model = CompositeObservationModel((gaussian_model, poisson_model))

# Materialize with data and hyperparameters
y_composite = CompositeObservations(([1.0, 2.0], [3, 4]))
composite_lik = composite_model(y_composite; σ=1.5)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/observation_models/composite/composite_observation_model.jl#L1-L22" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.CompositeLikelihood' href='#GaussianMarkovRandomFields.CompositeLikelihood'><span class="jlbinding">GaussianMarkovRandomFields.CompositeLikelihood</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
CompositeLikelihood{T<:Tuple} <: ObservationLikelihood
```


A materialized composite likelihood that combines multiple component likelihoods.

Created by calling a `CompositeObservationModel` with observation data and hyperparameters. Provides efficient evaluation of log-likelihood, gradient, and Hessian by summing contributions from all component likelihoods.

**Fields**
- `components::T`: Tuple of materialized component likelihoods
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/observation_models/composite/composite_observation_model.jl#L34-L45" target="_blank" rel="noreferrer">source</a></Badge>

</details>

