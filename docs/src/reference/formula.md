# Formula Interface

The formula interface lets you compose models succinctly using `StatsModels.jl` syntax.
It maps terms on the right-hand side to latent components and assembles a sparse
design matrix that connects observations to the combined latent field.
To use the formula syntax, load `StatsModels` first to activate the corresponding package extension.

## Quick Example

```julia
using GaussianMarkovRandomFields, StatsModels, Distributions, SparseArrays

# Suppose W is a spatial adjacency for regions; y are counts with an offset
besag = Besag(W)                         # structured spatial effect
iid = IID()                              # unstructured random effect
f = @formula(y ~ 1 + x + iid(region) + besag(region))
comp = build_formula_components(f, data; family = Poisson)
lik  = comp.obs_model(data.y; offset = data.logE)
prior = comp.combined_model(; τ_besag=1.0, τ_iid=1.0)
post  = gaussian_approximation(prior, lik)
```

## Specifying Constraints

Random effect terms support constraint specifications for identifiability and other modeling requirements. This is essential for models like BYM where both spatial and IID effects need sum-to-zero constraints.

### Basic Usage

Create functor instances with constraints before using them in formulas:

```julia
using GaussianMarkovRandomFields, StatsModels

# Data
data = (
    y = randn(20),
    hospital = repeat(["A", "B", "C", "D"], 5),
    time = collect(1:20)
)

# Create constrained functors
iid_sz = IID(constraint=:sumtozero)
ar1_sz = AR1(constraint=:sumtozero)
rw1 = RandomWalk()  # Built-in sum-to-zero

# Use in formula
comp = build_formula_components(
    @formula(y ~ 0 + iid_sz(hospital) + ar1_sz(time)),
    data;
    family = Normal
)

# Result is a ConstrainedGMRF
gmrf = comp.combined_model(τ_iid=1.0, τ_ar1=1.0, ρ_ar1=0.7)
```

### BYM Model with Constraints

The BYM (Besag-York-Mollié) model uses sum-to-zero constraints for identifiability:

```julia
# Spatial adjacency matrix
W = sparse([...])

# Both components need sum-to-zero constraints
besag = Besag(W)                      # Spatial (built-in constraint)
iid_sz = IID(constraint=:sumtozero)   # Unstructured (explicit constraint)

comp = build_formula_components(
    @formula(y ~ 1 + besag(region) + iid_sz(region)),
    data;
    family = Poisson
)
```

### Custom Constraints

Specify custom linear constraints as `(A, e)` tuples where `Ax = e`:

```julia
# Constraint: first two groups have equal effects (x₁ = x₂)
A = [1.0 -1.0 0.0 0.0]  # x₁ - x₂ = 0
e = [0.0]

iid_custom = IID(constraint=(A, e))
comp = build_formula_components(@formula(y ~ 0 + iid_custom(group)), data; family=Normal)
```

### Constraint Support by Term

- **`IID(constraint=...)`**: Optional `constraint` parameter
  - `nothing` (default): unconstrained
  - `:sumtozero`: sum-to-zero constraint
  - `(A, e)`: custom constraint matrix and vector

- **`AR1(constraint=...)`**: Same as IID

- **`RandomWalk(additional_constraints=...)`**:
  - Always has built-in sum-to-zero
  - Optional `additional_constraints` for extra constraints beyond sum-to-zero

- **`Besag(...)`**: Always has built-in sum-to-zero constraints per connected component

### Unconstrained vs. Constrained

```julia
# Unconstrained
iid = IID()
comp1 = build_formula_components(@formula(y ~ 0 + iid(group)), data; family=Normal)
gmrf1 = comp1.combined_model(τ_iid=1.0)  # Standard GMRF

# Constrained
iid_sz = IID(constraint=:sumtozero)
comp2 = build_formula_components(@formula(y ~ 0 + iid_sz(group)), data; family=Normal)
gmrf2 = comp2.combined_model(τ_iid=1.0)  # ConstrainedGMRF
```

## Terms and Builders

```@docs
GaussianMarkovRandomFields.IID
GaussianMarkovRandomFields.RandomWalk
GaussianMarkovRandomFields.AR1
GaussianMarkovRandomFields.Besag
GaussianMarkovRandomFields.build_formula_components
```

## See Also

- The BYM + fixed effects Poisson tutorial shows this in practice: [Advanced GMRF modelling for disease mapping](@ref)

