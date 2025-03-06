# Spatial and spatiotemporal discretizations
## Discretizing SPDEs
```@docs
discretize
```

## Spatial discretization: FEM
```@docs
FEMDiscretization
ndim
Ferrite.ndofs(::FEMDiscretization)
```

```@docs
evaluation_matrix
node_selection_matrix
derivative_matrices
second_derivative_matrices
```

### Utilities
```@docs
assemble_mass_matrix
assemble_diffusion_matrix
assemble_advection_matrix
lump_matrix
assemble_streamline_diffusion_matrix
apply_soft_constraints!
```

## Temporal discretization and state-space models
```@docs
JointSSMMatrices
joint_ssm
ImplicitEulerSSM
ImplicitEulerJointSSMMatrices
```
