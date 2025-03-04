<h1 align="center">
  GMRFs.jl
</h1>

<p align="center">
    <picture align="center">
        <img alt="Logo for the GMRFs.jl package." src="https://github.com/timweiland/GMRFs.jl/blob/main/docs/src/assets/logo.svg" width="200px" height="200px">
    </picture>
    <br>
    <strong>⚡ Fast, flexible and user-centered Julia package for Bayesian inference with sparse Gaussians</strong>
</p>

<div align="center">

[![Build Status](https://github.com/timweiland/GMRFs.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/timweiland/GMRFs.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/timweiland/GMRFs.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/timweiland/GMRFs.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

</div>

## Contents

- [Installation](#installation)
- [Your first GMRF](#your-first-gmrf)
- [Contributing](#contributing)

## Installation

GMRFs.jl is not yet a registered Julia package.
Until it is, you can install it locally.
To do so:

1. [Download Julia (>= version 1.10)](https://julialang.org/downloads/).

2. Launch the Julia REPL and type `] add /path/to/GMRFs.jl`. 

## Your first GMRF

Let's construct a GMRF approximation to a Matern process on a square grid:

``` julia
using GMRFs

# Define mesh via Ferrite.jl
using Ferrite
grid = generate_grid(Triangle, (50, 50))
interpolation_fn = Lagrange{RefTriangle, 1}()
quad_rule = QuadratureRule{RefTriangle}(2)
disc = FEMDiscretization(grid, interpolation_fn, quad_rule)

# Define SPDE and discretize to get a GMRF
spde = MaternSPDE{2}(range = 0.3, smoothness = 1)
x = discretize(spde, disc)
```

`x` is a Gaussian distribution, and we can compute all the things Gaussians are
known for.

```julia
# Get interesting quantities
using Distributions
μ = mean(x)
σ_marginal = std(x)
samp = rand(x) # Sample
Q_linmap = precision_map(x) # Linear map
Q = to_matrix(Q_linmap) # Sparse matrix

# Form posterior under point observations
A = evaluation_matrix(disc, [Tensors.Vec(0.1, 0.0), Tensors.Vec(-0.3, 0.55)])
noise_precision = 1e2 # Inverse of the noise variance
y = [0.83, 0.12]
x_cond = condition_on_observations(x, A, noise_precision, y) # Again a GMRF!
```

Make sure to check the documentation for further examples!

## Contributing

TODO
