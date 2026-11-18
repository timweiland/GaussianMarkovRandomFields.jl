# Spatial and spatiotemporal discretizations
## Discretizing SPDEs
```@docs
discretize
```

## Spatial discretization: FEM
```@docs
FEMDiscretization
ndim
intrinsic_dim
Ferrite.ndofs(::FEMDiscretization)
```

```@docs
evaluation_matrix
node_selection_matrix
derivative_matrices
second_derivative_matrices
```

### Utilities

The FEM assembly helpers (mass, diffusion and advection matrices, lumping, soft
constraints) are internals of the FEM extension; see [Internals](@ref).

## Temporal discretization and state-space models

`discretize` builds spatiotemporal GMRFs from implicit-Euler state-space models. Their
building blocks (`joint_ssm`, `ImplicitEulerSSM`, ...) are internals; see [Internals](@ref).

