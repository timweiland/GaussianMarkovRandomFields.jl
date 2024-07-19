# GMRFs.jl

*Gaussian Markov Random Fields in Julia.*

## Introduction
`GMRFs.jl` provides utilities for working with Gaussian Markov Random Fields (GMRFs) in Julia.
The goal is to enable **flexible** and **efficient** Bayesian inference from GMRFs.
In particular, we support the creation of GMRFs through Finite Element discretizations of
Stochastic Partial Differential Equations (SPDEs).
This unlocks efficient GMRF-based approximations to commonly used Gaussian process (GP)
priors [1].
Because of the expressive power of SPDEs, it also lets you use fairly problem-tailored
priors [2].

To get started with `GMRFs.jl`, consider going through the Tutorials.

## Installation
`GMRFs.jl` can be installed from the Pkg REPL (press `]` in the Julia REPL):

```
pkg> add GMRFs
```

Afterwards, you may load the package using

```julia
using GMRFs
```

You're good to go!
