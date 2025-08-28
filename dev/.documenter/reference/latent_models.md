
# Latent Models {#Latent-Models}

Latent models provide a structured way to define commonly used Gaussian Markov Random Fields (GMRFs) by specifying their key components: hyperparameters, precision matrices, mean vectors, and constraints. This abstraction enables easy construction and combination of standard models used in spatial statistics, time series analysis, and Bayesian modeling.

## Overview {#Overview}

The `LatentModel` interface standardizes GMRF construction through a simple pattern:
1. **Define the model structure** (e.g., temporal, spatial, independence)
  
2. **Specify hyperparameters** and their validation
  
3. **Construct GMRFs** with automatic constraint handling
  

```julia
using GaussianMarkovRandomFields

# Define a temporal AR1 model
ar1 = AR1Model(100)
hyperparameters(ar1)  # (τ = Real, ρ = Real)

# Construct GMRF with specific parameters
gmrf = ar1(τ=2.0, ρ=0.8)  # Returns GMRF or ConstrainedGMRF automatically
```


## LatentModel Interface {#LatentModel-Interface}
<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.LatentModel' href='#GaussianMarkovRandomFields.LatentModel'><span class="jlbinding">GaussianMarkovRandomFields.LatentModel</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
LatentModel
```


Abstract type for latent variable models that can be used to construct GMRFs.

A LatentModel provides a structured way to define commonly used GMRFs such as  AR1, RW1, and other temporal/spatial models by specifying:
1. The hyperparameters of the model
  
2. How to construct the precision matrix from hyperparameters  
  
3. How to construct the mean vector from hyperparameters
  
4. Any linear constraints that should be applied
  

**Interface**

Each concrete subtype must implement:
- `length(model)`: Return the size/dimension of the latent process
  
- `hyperparameters(model)`: Return a NamedTuple describing the hyperparameters
  
- `precision_matrix(model; kwargs...)`: Construct precision matrix from hyperparameter values
  
- `mean(model; kwargs...)`: Construct mean vector from hyperparameter values  
  
- `constraints(model; kwargs...)`: Return constraint information or `nothing`
  
- `model_name(model)`: Return a Symbol representing the preferred name for this model type
  
- `(model)(; kwargs...)`: Instantiate a concrete GMRF from hyperparameter values
  

**Usage**

```julia
# Define a model
model = SomeLatentModel(n=100)

# Get hyperparameter specification
params = hyperparameters(model)  # e.g. (τ=Real, ρ=Real)

# Instantiate GMRF with specific parameter values
gmrf = model(τ=2.0, ρ=0.8)  # Returns GMRF or ConstrainedGMRF
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/latent_models/latent_model.jl#L3-L39" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.hyperparameters-Tuple{LatentModel}' href='#GaussianMarkovRandomFields.hyperparameters-Tuple{LatentModel}'><span class="jlbinding">GaussianMarkovRandomFields.hyperparameters</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
hyperparameters(model::LatentModel)
```


Return a NamedTuple describing the hyperparameters and their types for the model.

**Returns**

A NamedTuple where keys are parameter names and values are their expected types.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/latent_models/latent_model.jl#L54-L61" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.precision_matrix-Tuple{LatentModel}' href='#GaussianMarkovRandomFields.precision_matrix-Tuple{LatentModel}'><span class="jlbinding">GaussianMarkovRandomFields.precision_matrix</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
precision_matrix(model::LatentModel; kwargs...)
```


Construct the precision matrix for the model given hyperparameter values.

**Arguments**
- `model`: The LatentModel instance
  
- `kwargs...`: Hyperparameter values as keyword arguments
  

**Returns**

A precision matrix (AbstractMatrix or LinearMap) for use in GMRF construction.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/latent_models/latent_model.jl#L66-L77" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='Statistics.mean' href='#Statistics.mean'><span class="jlbinding">Statistics.mean</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
mean(model::LatentModel; kwargs...)
```


Construct the mean vector for the model given hyperparameter values.

**Arguments**
- `model`: The LatentModel instance  
  
- `kwargs...`: Hyperparameter values as keyword arguments
  

**Returns**

A mean vector (AbstractVector) for use in GMRF construction.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/latent_models/latent_model.jl#L82-L93" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.constraints' href='#GaussianMarkovRandomFields.constraints'><span class="jlbinding">GaussianMarkovRandomFields.constraints</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
constraints(model::LatentModel; kwargs...)
```


Return constraint information for the model given hyperparameter values.

**Arguments**
- `model`: The LatentModel instance
  
- `kwargs...`: Hyperparameter values as keyword arguments  
  

**Returns**

Either `nothing` if no constraints, or a tuple `(A, e)` where `A` is the  constraint matrix and `e` is the constraint vector such that `Ax = e`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/latent_models/latent_model.jl#L98-L110" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.model_name' href='#GaussianMarkovRandomFields.model_name'><span class="jlbinding">GaussianMarkovRandomFields.model_name</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
model_name(model::LatentModel)
```


Return a Symbol representing the preferred name for this model type.

This name is used for parameter prefixing in CombinedModel to avoid conflicts. For example, if two models both have a τ parameter, they become τ_ar1, τ_besag, etc.

**Returns**

A Symbol that will be used as the suffix in parameter names.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/latent_models/latent_model.jl#L115-L125" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Available Models {#Available-Models}

### Temporal Models {#Temporal-Models}
<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.AR1Model' href='#GaussianMarkovRandomFields.AR1Model'><span class="jlbinding">GaussianMarkovRandomFields.AR1Model</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
AR1Model(n::Int)
```


A first-order autoregressive (AR1) latent model for constructing AR1 GMRFs.

The AR1 model represents a temporal process where each observation depends on  the previous observation with some correlation ρ and precision τ.

**Mathematical Description**

For n observations, the AR1 process has:
- Zero mean: μ = 0
  
- Precision matrix Q with tridiagonal structure:
  - Q[1,1] = τ
    
  - Q[i,i] = (1 + ρ²)τ for i = 2,...,n-1  
    
  - Q[n,n] = τ
    
  - Q[i,i+1] = Q[i+1,i] = -ρτ for i = 1,...,n-1
    
  

**Hyperparameters**
- `τ`: Precision parameter (τ &gt; 0)
  
- `ρ`: Correlation parameter (|ρ| &lt; 1)
  

**Fields**
- `n::Int`: Length of the AR1 process
  

**Example**

```julia
model = AR1Model(100)
gmrf = model(τ=2.0, ρ=0.8)  # Construct AR1 GMRF
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/latent_models/ar1.jl#L6-L36" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.RW1Model' href='#GaussianMarkovRandomFields.RW1Model'><span class="jlbinding">GaussianMarkovRandomFields.RW1Model</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
RW1Model(n::Int)
```


A first-order random walk (RW1) latent model for constructing intrinsic GMRFs.

The RW1 model represents a non-stationary temporal process where each observation  is the previous observation plus Gaussian noise. This creates a smooth trend model that&#39;s popular for temporal smoothing and time-varying effects.

**Mathematical Description**

The RW1 process defines increments: x[i+1] - x[i] ~ N(0, τ⁻¹) for i = 1,...,n-1. This leads to a singular precision matrix with the tridiagonal structure:
- Q[1,1] = 1, Q[n,n] = 1  
  
- Q[i,i] = 2 for i = 2,...,n-1
  
- Q[i,i+1] = Q[i+1,i] = -1 for i = 1,...,n-1
  

Since this matrix is singular (rank n-1), we handle it as an intrinsic GMRF by:
1. Scaling by τ first, then adding small regularization (1e-5) to diagonal for numerical stability
  
2. Adding sum-to-zero constraint: sum(x) = 0
  

**Hyperparameters**
- `τ`: Precision parameter (τ &gt; 0)
  

**Fields**
- `n::Int`: Length of the RW1 process
  
- `regularization::Float64`: Small value added to diagonal after scaling (default 1e-5)
  

**Example**

```julia
model = RW1Model(100)
gmrf = model(τ=1.0)  # Returns ConstrainedGMRF with sum-to-zero constraint
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/latent_models/rw1.jl#L6-L39" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Spatial Models {#Spatial-Models}
<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.BesagModel' href='#GaussianMarkovRandomFields.BesagModel'><span class="jlbinding">GaussianMarkovRandomFields.BesagModel</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
BesagModel(adjacency::AbstractMatrix; regularization::Float64 = 1e-5)
```


A Besag model for spatial latent effects on graphs using intrinsic Conditional Autoregressive (CAR) structure.

The Besag model represents spatial dependence where each node&#39;s precision depends on its graph neighbors. This creates a graph Laplacian precision structure that&#39;s widely used for spatial smoothing on irregular lattices.

**Mathematical Description**

For a graph with adjacency matrix W, the precision matrix follows:
- Q[i,j] = -τ if nodes i and j are neighbors (W[i,j] = 1)
  
- Q[i,i] = τ * degree[i] where degree[i] = sum(W[i,:])  
  
- All other entries are 0
  

Since this matrix is singular (rank n-1), we handle it as an intrinsic GMRF by:
1. Scaling by τ first, then adding small regularization (1e-5) to diagonal for numerical stability
  
2. Adding sum-to-zero constraint: sum(x) = 0
  

**Hyperparameters**
- `τ`: Precision parameter (τ &gt; 0)
  

**Fields**
- `adjacency::M`: Adjacency matrix W (preserves input structure - sparse, SymTridiagonal, etc.)
  
- `regularization::Float64`: Small value added to diagonal after scaling (default 1e-5)
  

**Example**

```julia
# 4-node cycle graph - can use sparse, SymTridiagonal, or Matrix
W = sparse(Bool[0 1 0 1; 1 0 1 0; 0 1 0 1; 1 0 1 0])
model = BesagModel(W)
gmrf = model(τ=1.0)  # Returns ConstrainedGMRF with sum-to-zero constraint
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/latent_models/besag.jl#L6-L39" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Independence Models {#Independence-Models}
<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.IIDModel' href='#GaussianMarkovRandomFields.IIDModel'><span class="jlbinding">GaussianMarkovRandomFields.IIDModel</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
IIDModel(n::Int)
```


An independent and identically distributed (IID) latent model for constructing simple diagonal GMRFs.

The IID model represents independent Gaussian random variables with identical precision τ. This is the simplest possible latent model, equivalent to a scaled identity precision matrix.

**Mathematical Description**

Each element is independent: x[i] ~ N(0, τ⁻¹) for i = 1,...,n. The precision matrix is simply: Q = τ * I(n)

This model is useful for:
- Modeling independent effects or noise
  
- Baseline comparisons with structured models
  
- Teaching/demonstration purposes
  

**Hyperparameters**
- `τ`: Precision parameter (τ &gt; 0)
  

**Fields**
- `n::Int`: Length of the IID process
  

**Example**

```julia
model = IIDModel(100)
gmrf = model(τ=2.0)  # Returns GMRF with precision 2.0 * I(100)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/latent_models/iid.jl#L6-L35" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Model Composition {#Model-Composition}
<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.CombinedModel' href='#GaussianMarkovRandomFields.CombinedModel'><span class="jlbinding">GaussianMarkovRandomFields.CombinedModel</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
CombinedModel(components::Vector{<:LatentModel})
```


A combination of multiple LatentModel instances into a single block-structured GMRF.

This enables modeling with multiple latent components, such as the popular BYM model  (Besag-York-Mollié) which combines spatial (Besag) and independent (IID) effects.

**Mathematical Description**

Given k component models with sizes n₁, n₂, ..., nₖ:
- Combined precision matrix: Q = blockdiag(Q₁, Q₂, ..., Qₖ)
  
- Combined mean vector: μ = vcat(μ₁, μ₂, ..., μₖ)  
  
- Combined constraints: Block-diagonal constraint structure preserving individual constraints
  

**Parameter Naming**

To avoid conflicts when multiple models have the same hyperparameters (e.g., multiple τ), parameters are automatically prefixed with model names:
- Single occurrence: τ_besag, ρ_ar1
  
- Multiple occurrences: τ_besag, τ_besag_2, τ_besag_3
  

**Fields**
- `components::Vector{<:LatentModel}`: The individual latent models  
  
- `component_sizes::Vector{Int}`: Cached sizes of each component
  
- `total_size::Int`: Total size of the combined model
  

**Example - BYM Model**

```julia
# BYM model: spatial Besag + independent IID effects
W = sparse(adjacency_matrix)  # Spatial adjacency
besag = BesagModel(W)         # Spatial component  
iid = IIDModel(n)            # Independent component

# Vector constructor
bym = CombinedModel([besag, iid])
# Or variadic constructor (syntactic sugar)
bym = CombinedModel(besag, iid)

# Usage with automatically prefixed parameters
gmrf = bym(τ_besag=1.0, τ_iid=2.0)
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/latent_models/combined.jl#L6-L48" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Usage Examples {#Usage-Examples}

### Basic Model Construction {#Basic-Model-Construction}

```julia
# Temporal AR1 process
ar1 = AR1Model(100)
gmrf = ar1(τ=2.0, ρ=0.8)

# Spatial Besag model
W = sparse_adjacency_matrix
besag = BesagModel(W)
gmrf = besag(τ=1.0)  # Returns ConstrainedGMRF with sum-to-zero constraint
```


### Model Composition {#Model-Composition-2}

```julia
# Classic BYM model: spatial + independent effects  
W = spatial_adjacency_matrix
bym = CombinedModel(BesagModel(W), IIDModel(n))

# Check combined hyperparameters
hyperparameters(bym)  # (τ_besag = Real, τ_iid = Real)

# Construct combined GMRF
gmrf = bym(τ_besag=1.0, τ_iid=2.0)
```


### Smart Parameter Naming {#Smart-Parameter-Naming}

CombinedModel automatically handles parameter naming conflicts:

```julia
# Single occurrence: no suffix
CombinedModel(AR1Model(10), BesagModel(W))  # τ_ar1, ρ_ar1, τ_besag

# Multiple occurrences: numbered suffixes  
CombinedModel(IIDModel(5), IIDModel(10), IIDModel(15))  # τ_iid, τ_iid_2, τ_iid_3
```


### Advanced Compositions {#Advanced-Compositions}

```julia
# Spatiotemporal model with three components
W = spatial_adjacency(n_regions)
model = CombinedModel(
    BesagModel(W),           # Spatial structure
    RW1Model(n_time),        # Temporal trend  
    IIDModel(n_regions * n_time)  # Independent effects
)

gmrf = model(τ_besag=1.0, τ_rw1=2.0, τ_iid=0.1)
```


## Key Features {#Key-Features}

### Automatic Constraint Handling {#Automatic-Constraint-Handling}

Models automatically determine GMRF type based on constraints:

```julia
ar1 = AR1Model(10)     # No constraints → GMRF
rw1 = RW1Model(10)     # Sum-to-zero constraint → ConstrainedGMRF
mixed = CombinedModel(ar1, rw1)  # Inherits constraints → ConstrainedGMRF
```


### Parameter Validation {#Parameter-Validation}

All models include built-in parameter validation:

```julia
ar1 = AR1Model(100)
ar1(τ=-1.0, ρ=0.8)   # ArgumentError: τ must be positive
ar1(τ=1.0, ρ=1.5)    # ArgumentError: |ρ| must be < 1
```


### Efficient Sparse Operations {#Efficient-Sparse-Operations}
- Block-diagonal precision matrices are always sparse
  
- Input matrix structures (sparse, SymTridiagonal) are preserved
  
- Minimal memory allocations through size caching
  

## See Also {#See-Also}
- [`GMRF`](/reference/gmrfs#GaussianMarkovRandomFields.GMRF) and [`ConstrainedGMRF`](/reference/hard_constraints#GaussianMarkovRandomFields.ConstrainedGMRF) for the underlying GMRF types
  
- [Hard Constraints](/reference/hard_constraints#Hard-Constraints) for constraint handling details
  
- [Observation Models](/reference/observation_models#Observation-Models) for linking GMRFs to data
  
