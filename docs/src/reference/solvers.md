# Solvers

Fundamentally, all interesting quantities of GMRFs (samples, marginal variances,
posterior means, ...) must be computed through **sparse linear algebra**.
GaussianMarkovRandomFields.jl uses [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl) as the backend for all linear algebra operations, providing access to a wide range of modern solvers and preconditioners while maintaining a unified interface.
Our goal is to provide sane defaults that "just work" for most users, while
still allowing power users to customize the behavior through LinearSolve.jl algorithms.
Read further to learn about the available capabilities and how to use them.

## Direct Methods (Cholesky, LU)
[CHOLMOD](https://github.com/DrTimothyAldenDavis/SuiteSparse) and other direct solvers are 
state-of-the-art methods for sparse linear systems.
LinearSolve.jl automatically selects appropriate direct methods (typically Cholesky for symmetric positive definite systems like GMRF precision matrices) to compute factorizations of the precision matrix, which are then leveraged to draw samples and compute posterior means.
Marginal variances are computed using selected inversion when available, or RBMC as a fallback.

## Pardiso
[Pardiso](https://panua.ch/pardiso/) is a state-of-the-art direct solver for
sparse linear systems with excellent performance on large-dimensional GMRFs
and highly efficient methods for marginal variance computation.

To use Pardiso, you need to set up and load [Pardiso.jl](https://github.com/JuliaSparse/Pardiso.jl), then specify `alg=LinearSolve.PardisoJL()` when creating your GMRF:

```julia
using Pardiso, LinearSolve
gmrf = GMRF(μ, Q, alg=LinearSolve.PardisoJL())
```

See the [LinearSolve.jl documentation](https://docs.sciml.ai/LinearSolve/stable/solvers/solvers/#LinearSolve.PardisoJL) for more information about PardisoJL algorithm options.

## Iterative Methods (CG, GMRES)
The Conjugate Gradient (CG) method and other iterative methods are efficient approaches for solving large sparse symmetric positive-definite linear systems.
LinearSolve.jl provides access to various iterative solvers that can be used for large-scale GMRFs where direct methods become prohibitively expensive.

For symmetric systems like GMRFs, CG-based methods are typically most appropriate:

```julia
using LinearSolve
gmrf = GMRF(μ, Q, alg=LinearSolve.KrylovJL_CG())
```

## Variance Computation Strategies
Fundamentally, computing marginal variances of GMRFs is not trivial, as it
requires computing the diagonal entries of the covariance matrix (which is the
inverse of the precision matrix).

**Selected Inversion** (including Takahashi recursions) is a highly accurate and stable method for computing these variances when available for the chosen algorithm. This is automatically used when supported.

When selected inversion is not available, the package automatically falls back to **RBMC** (Rao-Blackwellized Monte Carlo), a sampling-based approach that is less accurate but can be much faster for large GMRFs.

```@docs
AbstractVarianceStrategy
RBMCStrategy
BlockRBMCStrategy
```

## Advanced Operations

```@docs
logdet_cov
selinv
backward_solve
```
 

## Choosing a Solver
LinearSolve.jl automatically selects appropriate algorithms based on matrix properties:
- **Direct methods** (Cholesky, LU) for small to medium-sized GMRFs
- **Iterative methods** (CG, GMRES) for large sparse GMRFs where direct methods become prohibitively expensive

This matches our general recommendations:
Direct solvers combined with selected inversion are highly accurate and stable and should be used whenever possible.
However, direct solvers become prohibitively expensive for very large-scale GMRFs, both in terms of compute and memory use.
In these regimes, iterative methods may still be a viable option.

If you have access to both strong parallel computing resources and a Pardiso license, `LinearSolve.PardisoJL()` may provide the best performance.
