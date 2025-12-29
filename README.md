<h1 align="center">
  GaussianMarkovRandomFields.jl
</h1>

<p align="center">
    <picture align="center">
        <img alt="Logo for the GaussianMarkovRandomFields.jl package." src="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/main/docs/src/assets/logo.svg" width="200px" height="200px">
    </picture>
    <br>
    <strong>⚡ Fast, flexible and user-centered Julia package for Bayesian inference with sparse Gaussians</strong>
</p>

<div align="center">

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://timweiland.github.io/GaussianMarkovRandomFields.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://timweiland.github.io/GaussianMarkovRandomFields.jl/dev)

[![Build Status](https://github.com/timweiland/GaussianMarkovRandomFields.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/timweiland/GaussianMarkovRandomFields.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/timweiland/GaussianMarkovRandomFields.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/timweiland/GaussianMarkovRandomFields.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)
[![DOI](https://zenodo.org/badge/827801443.svg)](https://doi.org/10.5281/zenodo.18088215)

</div>

Gaussian Markov Random Fields (GMRFs) are Gaussian distributions with sparse
precision (inverse covariance) matrices.
GaussianMarkovRandomFields.jl provides utilities for working with GMRFs in Julia.
The goal is to enable **flexible** and **efficient** Bayesian inference from
GMRFs, powered by sparse linear algebra.

In particular, we support the creation of GMRFs through finite element method
discretizations of stochastic partial differential equations (SPDEs).
This unlocks efficient GMRF-based approximations to commonly used Gaussian
process priors.
Furthermore, the expressive power of SPDEs allows for flexible, problem-tailored
priors.

## Contents

- [Installation](#installation)
- [Your first GMRF](#your-first-gmrf)
- [Contributing](#contributing)

## Installation

GaussianMarkovRandomFields.jl is a registered Julia package.
To install it, launch the Julia REPL and type `] add GaussianMarkovRandomFields`. 

## Your first GMRF

Let's construct a GMRF approximation to a Matérn process from observation points:

``` julia
using GaussianMarkovRandomFields

# Define observation points  
points = [0.1 0.0; -0.3 0.55; 0.2 0.8; -0.1 -0.2]  # N×2 matrix

# Create Matérn latent model (automatically generates mesh and discretization)
model = MaternModel(points; smoothness = 1)
x = model(range = 0.3)  # Construct GMRF with specified range
```

`x` is a Gaussian distribution, and we can compute all the things Gaussians are
known for.

```julia
# Get interesting quantities
μ = mean(x)
σ_marginal = std(x)
samp = rand(x)  # Sample
Q = precision_map(x)  # Sparse precision matrix

# Form posterior under point observations using new helpers
using Distributions: Normal
obs_model = PointEvaluationObsModel(model.discretization, points, Normal)
y = [0.83, 0.12, 0.45, -0.21]
obs_likelihood = obs_model(y; σ = 0.1)
x_cond = gaussian_approximation(x, obs_likelihood)  # Posterior GMRF!
```

Make sure to check the documentation for further examples!

## Contributing

Check our [contribution guidelines](./CONTRIBUTING.md).
