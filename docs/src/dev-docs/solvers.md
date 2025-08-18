# LinearSolve.jl Integration

## Overview

GaussianMarkovRandomFields.jl uses LinearSolve.jl as the backend for all linear algebra operations. This provides access to a wide range of solvers and preconditioners while maintaining a unified interface.

## Key Components

The LinearSolve.jl integration provides several specialized modules:

- **Selected Inversion**: Efficient computation of diagonal elements of matrix inverses
- **Backward Solve**: Specialized solve operations for GMRF computations  
- **Log Determinant**: Efficient computation of log determinants for sparse matrices
- **RBMC**: Rao-Blackwellized Monte Carlo for marginal variance estimation

## RBMC Fallback

When selected inversion is not available for a particular algorithm, the package automatically falls back to RBMC (Rao-Blackwellized Monte Carlo) for computing marginal variances:

```julia
# RBMC automatically used when selected inversion unavailable
variance_vec = var(gmrf)  # Uses RBMC if needed
```
