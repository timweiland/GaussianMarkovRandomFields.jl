```@raw html
---
layout: home

hero:
  name: "GMRFs in Julia"
  text: "Fast and Flexible Latent Gaussian Modelling"
  tagline: "Clean syntax, fast computations. Powered by the Julia ecosystem."
  image:
    src: /logo.svg
    alt: "GaussianMarkovRandomFields.jl"
  actions:
    - theme: brand
      text: "Get Started"
      link: /tutorials/getting_started
    - theme: alt
      text: "View on GitHub"
      link: https://github.com/timweiland/GaussianMarkovRandomFields.jl
    - theme: alt  
      text: "API Reference"
      link: /reference/

features:
  - title: "⚡ High Performance"
    details: "Leverages sparse matrix structures for fast computations, enabled by LinearSolve.jl and SelectedInversion.jl."
  - title: "📊 Smart Observation Models"
    details: "Built-in exponential family support. Sparse autodiff makes it easy to define your own likelihoods."
  - title: "🎯 Ready-to-Use Latent Models"
    details: "AR1, RW1, Besag, ... out-of-the-box. Combine them seamlessly to build complex hierarchical structures."
  - title: "🔬 SPDE Discretizations"
    details: "Model with a GP, compute with a GMRF. The SPDE approach makes it possible."
  - title: "🧮 Solver Variety"
    details: "CHOLMOD, Pardiso, Krylov methods... Your choice - we make it work."
  - title: "🔗 Composable Design"
    details: "Mix and match components: combine latent models, stack observation models, chain transformations - everything just works together."
---
```

## What are Gaussian Markov Random Fields?

A Gaussian Markov Random Field (GMRF) is a Gaussian distribution whose
*precision* matrix — the inverse of the covariance — is sparse.

Sparsity in the precision is a modelling statement. A zero in
entry $(i, j)$ of the precision matrix says precisely that $x_i$ and $x_j$ are
conditionally independent given all the other variables. Most quantities we
model in space and time behave that way: the temperature here depends on the
temperature next door, and only indirectly on the temperature a hundred
kilometres away. Encoding that structure in the precision matrix is what makes
the field *Markov*.

This brings nice computational benefits. The covariance matrix of such a distribution is
usually completely dense, so working with it directly costs $O(n^3)$ and becomes
hopeless as $n$ grows. The precision matrix is sparse, and sparse Cholesky
factorization takes advantage of that — which is how GMRFs reach problem sizes
where a naive Gaussian process cannot keep up.

The difficulty has always been that the interesting priors are awkward to write
down in precision form. One answer for spatial problems is the SPDE approach:
state the model as a stochastic partial differential equation, discretize it with
finite elements, and obtain a GMRF that approximates it.
[Getting started](@ref) walks through both a hand-written precision matrix and an SPDE-derived one.

## Quick Start

Install the package:

```julia
using Pkg
Pkg.add("GaussianMarkovRandomFields")
using GaussianMarkovRandomFields
```

## Related packages

[Latte.jl](https://lattejl.org) is a probabilistic programming language for
latent Gaussian models, built by the same author on top of this package. It
provides INLA, TMB-style Laplace approximations and HMC-Laplace behind a concise
model syntax. If you want a complete inference workflow rather than the
components to build one, start there; this package is what it uses underneath.

## Getting Help

Questions, bug reports and feature requests all belong in the
[issue tracker](https://github.com/timweiland/GaussianMarkovRandomFields.jl/issues).
Please open an issue rather than emailing the maintainer, so that answers stay
searchable for everyone.

The [contribution guidelines](https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/main/CONTRIBUTING.md)
describe what to include in an issue, how to contribute code, and what to expect
regarding response times and project scope.

## Citing

If you use GaussianMarkovRandomFields.jl in your research, please cite it via its
[Zenodo archive](https://doi.org/10.5281/zenodo.18088214). That DOI always
resolves to the latest release, and the Zenodo page offers BibTeX and other
export formats as well as per-version DOIs.

```bibtex
@software{weiland_gmrf_jl,
  author    = {Weiland, Tim},
  title     = {{GaussianMarkovRandomFields.jl}},
  publisher = {Zenodo},
  year      = {2025},
  doi       = {10.5281/zenodo.18088214},
  url       = {https://doi.org/10.5281/zenodo.18088214}
}
```
