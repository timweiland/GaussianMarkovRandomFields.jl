# Latent Models

Latent models provide a structured way to define commonly used Gaussian Markov Random Fields (GMRFs) by specifying their key components: hyperparameters, precision matrices, mean vectors, and constraints. This abstraction enables easy construction and combination of standard models used in spatial statistics, time series analysis, and Bayesian modeling.

## Overview

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

## LatentModel Interface

```@docs
LatentModel
hyperparameters(::LatentModel)
precision_matrix(::LatentModel)
mean
constraints
model_name
```

## Available Models

### Temporal Models

```@docs
AR1Model
RW1Model
```

### Spatial Models

```@docs
MaternModel
BesagModel
```

### Independence Models

```@docs
IIDModel
FixedEffectsModel
```

### Model Composition

```@docs
CombinedModel
```

## Usage Examples

### Basic Model Construction

```julia
# Temporal AR1 process
ar1 = AR1Model(100)
gmrf = ar1(τ=2.0, ρ=0.8)

# Spatial Matérn model from points
points = [0.0 0.0; 1.0 0.0; 0.5 1.0]  # N×2 matrix
matern = MaternModel(points; smoothness = 2)
gmrf = matern(range=1.5)

# Spatial Besag model
W = sparse_adjacency_matrix
besag = BesagModel(W)
gmrf = besag(τ=1.0)  # Returns ConstrainedGMRF with sum-to-zero constraint
```

### Model Composition

```julia
# Classic BYM model: spatial + independent effects  
W = spatial_adjacency_matrix
bym = CombinedModel(BesagModel(W), IIDModel(n))

# Check combined hyperparameters
hyperparameters(bym)  # (τ_besag = Real, τ_iid = Real)

# Construct combined GMRF
gmrf = bym(τ_besag=1.0, τ_iid=2.0)

# Continuous spatial field + independent effects
points = generate_observation_points()
spatial_matern = MaternModel(points; smoothness = 1) 
independent_effects = IIDModel(length(points))
combined = CombinedModel(spatial_matern, independent_effects)
gmrf = combined(range=2.0, τ_iid=0.1)
```

### Smart Parameter Naming

CombinedModel automatically handles parameter naming conflicts:

```julia
# Single occurrence: no suffix
CombinedModel(AR1Model(10), BesagModel(W))  # τ_ar1, ρ_ar1, τ_besag

# Multiple occurrences: numbered suffixes  
CombinedModel(IIDModel(5), IIDModel(10), IIDModel(15))  # τ_iid, τ_iid_2, τ_iid_3
```

### Advanced Compositions

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

## Key Features

### Automatic Constraint Handling

Models automatically determine GMRF type based on constraints:

```julia
ar1 = AR1Model(10)     # No constraints → GMRF
rw1 = RW1Model(10)     # Sum-to-zero constraint → ConstrainedGMRF
mixed = CombinedModel(ar1, rw1)  # Inherits constraints → ConstrainedGMRF
```

### Parameter Validation

All models include built-in parameter validation:

```julia
ar1 = AR1Model(100)
ar1(τ=-1.0, ρ=0.8)   # ArgumentError: τ must be positive
ar1(τ=1.0, ρ=1.5)    # ArgumentError: |ρ| must be < 1
```

### Efficient Sparse Operations

- Block-diagonal precision matrices are always sparse
- Input matrix structures (sparse, SymTridiagonal) are preserved
- Minimal memory allocations through size caching

## See Also

- [`GMRF`](@ref) and [`ConstrainedGMRF`](@ref) for the underlying GMRF types
- [Hard Constraints](@ref) for constraint handling details
- [Observation Models](@ref) for linking GMRFs to data

## Formula Terms

Prefer writing models with formulas? The latent components discussed here can be
constructed via formula terms and assembled automatically into a combined model
and design matrix. See the Formula Interface reference for details:

- [Formula Interface](@ref) — terms (`IID`, `RandomWalk`, `AR1`, `Besag`) and
  `build_formula_components`.
- For a worked example combining Besag + IID + fixed effects under a Poisson
  likelihood with offset, see the tutorial:
  [Advanced GMRF modelling for disease mapping](@ref)
