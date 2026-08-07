# Tutorials

These tutorials are self-contained. They do not build on one another, so the
sensible thing is to go straight to whichever one is closest to the problem you
have.

The exception is [Getting started](@ref), which introduces the three ideas that
all the others take for granted: latent models, observation models, and
posteriors. If you have not used the package before, read that one first —
everything else will make more sense afterwards.

## Start here

- [Getting started](@ref) runs through the whole workflow twice, once on a time
  series and once on a spatial field.

## Building models

- [Building autoregressive models](@ref) constructs an AR(1) model by hand, from
  its mean and precision matrix, and then again through the model interface.
  Read this if you want to see where the sparsity actually comes from.
- [Spatial Modelling with SPDEs](@ref) fits a Matérn field to zinc measurements
  in the soil near the Meuse river, and explains the link between Gaussian
  processes and SPDEs that makes this efficient.
- [Spatiotemporal Modelling with SPDEs](@ref) models a pollutant spreading
  through a river over time, on a 1D toy domain.
- [Modelling on manifolds](@ref) leaves flat Euclidean space behind: it builds a
  Matérn field on a sphere from a surface mesh, verifies it against the exact
  solution, and lets an advection–diffusion field rotate around the globe.
- [Boundary Conditions for SPDEs](@ref) explains why the default boundary
  behaviour of an SPDE discretization is often not what you want, and what to do
  about it.

## Non-Gaussian observations

- [Bernoulli Spatial Classification with a Matérn Field](@ref) builds a spatial
  classifier for binary labels on the Lansing Woods tree data.
- [Advanced GMRF modelling for disease mapping](@ref) works through a BYM model
  for Scottish lip cancer counts, combining a spatial effect with an
  unstructured one.

## Hyperparameter inference

- [Automatic Differentiation for GMRF Hyperparameters](@ref) computes gradients
  through GMRF operations, so that hyperparameters can be fitted by
  gradient-based optimization instead of being chosen by hand.
- [Automatic Differentiation and MCMC](@ref) goes a step further and samples the
  hyperparameter posterior with NUTS, via Turing.jl.

!!! tip "Going further"
    Both tutorials show the mechanics — gradients first, then samples. If you
    want the whole workflow rather than the building blocks,
    [Latte.jl](https://lattejl.org) is a probabilistic programming language for
    latent Gaussian models built on top of this package, providing INLA,
    TMB-style Laplace approximations and HMC-Laplace behind a model
    specification syntax. Its documentation has worked examples.

## Further topics

- [Reusing factorizations across hyperparameters](@ref) shows how to pay the
  symbolic factorization cost once instead of on every iteration of an inner
  loop.
- [KL-minimizing Sparse GMRF Approximations to Gaussian Processes](@ref)
  approximates a GP given by a kernel with a sparse GMRF directly, without
  passing through an SPDE.
- [Learning GMRFs from data with the graphical lasso](@ref) estimates a sparse
  precision matrix from sample data, for when the conditional independence
  structure is not known in advance.
