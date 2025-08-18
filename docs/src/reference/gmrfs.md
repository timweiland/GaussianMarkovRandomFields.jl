# GMRFs

```@docs
AbstractGMRF
GMRF
precision_map
precision_matrix
InformationVector
information_vector
```

## Metadata

```@docs
MetaGMRF
GMRFMetadata
```

## Arithmetic
```@docs
condition_on_observations
linear_condition
joint_gmrf
```

## Spatiotemporal setting
### Types
```@docs
AbstractSpatiotemporalGMRF
ImplicitEulerConstantMeshSTGMRF
ConcreteConstantMeshSTGMRF
```

### Quantities
```@docs
N_t
time_means
time_vars
time_stds
time_rands
discretization_at_time
```

### Utilities
```@docs
spatial_to_spatiotemporal
kronecker_product_spatiotemporal_model
```
