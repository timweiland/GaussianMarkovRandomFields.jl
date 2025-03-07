# GMRFs.jl

*Gaussian Markov Random Fields in Julia.*

## Introduction
Gaussian Markov Random Fields (GMRFs) are Gaussian distributions with sparse
precision (inverse covariance) matrices.
GMRFs.jl provides utilities for working with GMRFs in Julia.
The goal is to enable **flexible** and **efficient** Bayesian inference from
GMRFs, powered by sparse linear algebra.

In particular, we support the creation of GMRFs through finite element method
discretizations of stochastic partial differential equations (SPDEs).
This unlocks efficient GMRF-based approximations to commonly used Gaussian
process priors.
Furthermore, the expressive power of SPDEs allows for flexible, problem-tailored
priors.

To get started with GMRFs.jl, consider going through the [Tutorials](@ref).

## Installation
GMRFs.jl can be installed via its GitHub repo from the Pkg REPL:

```
pkg> add https://github.com/timweiland/GMRFs.jl
```

Afterwards, you may load the package using

```julia
using GMRFs
```

You're good to go!

!!! info
    While a fair amount of time was spent on this documentation, it is far from
    perfect. 
    If you spot parts of the documentation that you find confusing or that are
    incomplete, please open an issue or a pull request.
    Your help is much appreciated!

!!! tip
    Missing a feature?
    Let us know!
    If you're interested in contributing, that's even better!
    Check our contribution guidelines for assistance.
