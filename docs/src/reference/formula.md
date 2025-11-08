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

# Classic BYM: Both components need sum-to-zero constraints
besag = Besag(W)                      # Spatial (built-in constraint)
iid_sz = IID(constraint=:sumtozero)   # Unstructured (explicit constraint)

comp = build_formula_components(
    @formula(y ~ 1 + besag(region) + iid_sz(region)),
    data;
    family = Poisson
)
```

### BYM2 Model (Recommended)

The BYM2 model offers improved parameterization with a single precision τ and mixing parameter φ:

```julia
# BYM2 without intercept (no constraint needed)
bym2 = BYM2(W)
comp = build_formula_components(
    @formula(y ~ 0 + x + bym2(region)),
    data;
    family = Poisson
)
gmrf = comp.combined_model(τ_bym2=1.0, φ_bym2=0.5)

# BYM2 with intercept (constrain IID component for identifiability)
bym2_constrained = BYM2(W; iid_constraint=:sumtozero)
comp = build_formula_components(
    @formula(y ~ 1 + x + bym2_constrained(region)),
    data;
    family = Poisson
)
gmrf = comp.combined_model(τ_bym2=1.0, φ_bym2=0.5)
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

- **`BYM2(W; iid_constraint=..., additional_constraints=...)`**:
  - Spatial component: always has built-in sum-to-zero per connected component
  - IID component: optional `iid_constraint` parameter (default: unconstrained)
    - Use `iid_constraint=:sumtozero` when including a fixed intercept for identifiability
  - Optional `additional_constraints` for spatial component (beyond sum-to-zero)

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

## Separable Models (Kronecker Products)

For multi-dimensional processes with separable structure, use the `Separable` functor to compose
multiple random effects via Kronecker products. This is particularly useful for space-time models
where the precision matrix is `Q = Q_time ⊗ Q_space` with space varying fastest.

### Basic Space-Time Example

```julia
using GaussianMarkovRandomFields, StatsModels, SparseArrays

# Create data with space-time structure
n_time, n_space = 10, 5
data = (
    y = randn(n_time * n_space),
    time = repeat(1:n_time, outer=n_space),
    region = repeat(1:n_space, inner=n_time),
)

# Spatial adjacency (e.g., a chain or grid)
W = spzeros(n_space, n_space)
for i in 1:(n_space-1)
    W[i, i+1] = 1
    W[i+1, i] = 1
end

# Create separable model: Q = Q_time ⊗ Q_space
# RW1 for temporal smoothing, Besag for spatial smoothing
rw1 = RandomWalk()         # Temporal: RW1 with built-in sum-to-zero
besag = Besag(W)           # Spatial: Besag (intrinsic CAR)
st = Separable(rw1, besag) # Separable: Kronecker product

# Build formula
comp = build_formula_components(
    @formula(y ~ 1 + st(time, region)),
    data;
    family = Normal
)

# Instantiate GMRF
gmrf = comp.combined_model(τ_rw1_separable=1.0, τ_besag_separable=2.0)
```

The resulting precision matrix has a block-tridiagonal structure:
```
Q = Q_time ⊗ Q_space = [
    Q_space  -Q_space     0        0     ...
    -Q_space  2Q_space  -Q_space    0     ...
       0     -Q_space  2Q_space  -Q_space ...
       ...
]
```

### Constraint Composition

Constraints from each component are automatically composed and redundant constraints removed:

```julia
# Both RW1 (sum-to-zero in time) and Besag (sum-to-zero in space)
# produce constraints that are automatically handled
rw1 = RandomWalk()
besag = Besag(W)
st = Separable(rw1, besag)

comp = build_formula_components(@formula(y ~ 0 + st(time, region)), data; family=Normal)

# The resulting GMRF handles both constraints correctly with redundancy removal
gmrf = comp.combined_model(τ_rw1_separable=1.0, τ_besag_separable=1.0)
```

### N-way Separable Models

Extend beyond 2D with three or more components:

```julia
# 3D space-time-group model: Q = Q_time ⊗ Q_space ⊗ Q_group
n_group = 3
data_3d = (
    y = randn(n_time * n_space * n_group),
    time = repeat(repeat(1:n_time, outer=n_space), outer=n_group),
    region = repeat(repeat(1:n_space, inner=n_time), outer=n_group),
    group = repeat(1:n_group, inner=n_time*n_space),
)

rw1 = RandomWalk()
besag = Besag(W)
iid_group = IID()

# Separable with 3 components
sep3 = Separable(rw1, besag, iid_group)

comp = build_formula_components(
    @formula(y ~ 0 + sep3(time, region, group)),
    data_3d;
    family = Normal
)

gmrf = comp.combined_model(
    τ_rw1_separable = 1.0,
    τ_besag_separable = 1.0,
    τ_iid_separable = 0.5
)
```

### Component Ordering

The order of components in `Separable` determines the Kronecker order:
- `Separable(rw1, besag)` → Q = Q_rw1 ⊗ Q_besag (besag varies fastest)
- `Separable(besag, rw1)` → Q = Q_besag ⊗ Q_rw1 (rw1 varies fastest)

This affects the vectorization pattern but not which observations map to which latent states
(the indicator matrix handles that mapping appropriately).

### Hyperparameter Naming

Separable models use prefixed hyperparameter names for clarity:

```julia
# For Separable(rw1, besag):
# - τ_rw1_separable: precision for temporal component
# - τ_besag_separable: precision for spatial component

gmrf = comp.combined_model(τ_rw1_separable=1.0, τ_besag_separable=2.0)
```

When there are multiple components of the same type, indices are added:

```julia
rw1_a = RandomWalk()
rw1_b = RandomWalk()
sep = Separable(rw1_a, rw1_b)

# Hyperparameters are:
# - τ_rw1_separable: first RW1
# - τ_rw1_2_separable: second RW1

gmrf = comp.combined_model(τ_rw1_separable=1.0, τ_rw1_2_separable=1.5)
```

## Terms and Builders

```@docs
GaussianMarkovRandomFields.IID
GaussianMarkovRandomFields.RandomWalk
GaussianMarkovRandomFields.AR1
GaussianMarkovRandomFields.Besag
GaussianMarkovRandomFields.BYM2
GaussianMarkovRandomFields.Separable
GaussianMarkovRandomFields.build_formula_components
```

## See Also

- The BYM + fixed effects Poisson tutorial shows this in practice: [Advanced GMRF modelling for disease mapping](@ref)

