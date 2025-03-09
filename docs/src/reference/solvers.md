# Solvers

Fundamentally, all interesting quantities of GMRFs (samples, marginal variances,
posterior means, ...) must be computed through **sparse linear algebra**.
GaussianMarkovRandomFields.jl provides a number of so-called *solvers* which perform the underlying
computations in different ways and with different trade-offs.
Our goal is to provide sane defaults that "just work" for most users, while
still allowing power users to customize the behavior of the solvers.
Read further to learn about the available solvers and how to use them.

If you don't have time for the details and you're just looking for a
recommendation, skip to [Choosing a solver](#Choosing-a-solver).

If you're interested in the interface of solvers, skip to [Solver interface](#Solver-interface).

## Cholesky (CHOLMOD)
[CHOLMOD](https://github.com/DrTimothyAldenDavis/SuiteSparse) is a
state-of-the-art direct solver for sparse linear systems.
`CholeskySolver` uses CHOLMOD to compute the sparse Cholesky factorization of
the precision matrix, which it then leverages to draw samples and compute 
posterior means.
Marginal variances are computed using a user-specified variance strategy, which
defaults to a sampling-based approach.

```@docs
CholeskySolverBlueprint
```

## Pardiso
[Pardiso](https://panua.ch/pardiso/) is a state-of-the-art direct solver for
sparse linear systems.
Its main benefit over the `CholeskySolver` is its potential for better
performance on large-dimensional GMRFs, as well as its highly efficient
method for marginal variance computation.

To use this solver, you need to set up and load
[Pardiso.jl](https://github.com/JuliaSparse/Pardiso.jl).
The code for our Pardiso solver will then be loaded automatically through
a package extension.

```@docs
PardisoGMRFSolverBlueprint
```

## CG
The Conjugate Gradient (CG) method is an iterative method for solving symmetric
positive-definite linear systems.
`CGSolver` uses CG to compute posterior means and draw samples.

```@docs
CGSolverBlueprint
```

## Variance strategies
Fundamentally, computing marginal variances of GMRFs is not trivial, as it
requires computing the diagonal entries of the covariance matrix (which is the
inverse of the precision matrix).
The Takahashi recursions (sometimes referred to as a *sparse (partial) inverse*
method) are a highly accurate and stable method for computing these variances.
`TakahashiStrategy` uses the Takahashi recursions to compute marginal variances.
`PardisoSolver` also uses this algorithm internally, but with a highly optimized
implementation.

`RBMCStrategy` is a sampling-based approach to computing marginal variances.
It is less accurate than the Takahashi recursions, but can be much faster for
large GMRFs.

```@docs
TakahashiStrategy
RBMCStrategy
BlockRBMCStrategy
```
 

## Choosing a solver
The default solver uses the `CholeskySolver` for small to medium-sized GMRFs
and switches to the `CGSolver` for large GRMFs. 
Similarly, the default variance strategy is `TakahashiStrategy` for small to 
medium-sized GMRFs and `RBMCStrategy` for large GMRFs.
See:
```@docs
DefaultSolverBlueprint
```

Indeed, this matches our general recommendations:
Direct solvers combined with the Takahashi recursions are highly accurate and
stable and should be used whenever possible.
However, direct solvers become prohibitively expensive for very large-scale
GMRFs, both in terms of compute and memory use.
In these regimes, CG with a good preconditioner may still be a viable option.

If you have access to both strong parallel computing resources and a Pardiso
license, we recommend the use of the `PardisoGMRFSolver`.
In particular, Pardiso's Takahashi recursions are highly optimized and much
faster than the implementation used for our `TakahashiStrategy`.

## Solver interface
```@docs
AbstractSolver
AbstractSolverBlueprint
AbstractVarianceStrategy
gmrf_precision
compute_mean
compute_variance
compute_rand!
```
