# Solver Interface

## Constructing Solvers

```@docs
GaussianMarkovRandomFields.construct_solver
GaussianMarkovRandomFields.construct_conditional_solver
GaussianMarkovRandomFields.postprocess!
```

## Solver Access

All GMRF types store their solver as a direct field:

```julia
# Access the solver instance
solver = gmrf.solver

# Use solver methods directly
mean_vec = compute_mean(solver)
variance_vec = compute_variance(solver)
sample_vec = compute_rand!(solver, rng, zeros(length(gmrf)))
```

## Solver Construction Process

When constructing a GMRF, the solver is automatically created using:

1. `construct_solver(blueprint, mean, precision)` - creates the solver instance
2. `postprocess!(solver, gmrf)` - allows the solver to optimize itself using the full GMRF context

This two-step process ensures solvers can set up specialized preconditioners or other optimizations that require knowledge of the complete GMRF structure.
