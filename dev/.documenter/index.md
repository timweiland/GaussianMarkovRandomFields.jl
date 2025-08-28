---
layout: home

hero:
  name: "GMRFs in Julia"
  text: "Fast and Flexible Latent Gaussian Modelling"
  tagline: "Clean syntax, fast computations. Powered by the Julia ecosystem."
  image:
    src: /assets/logo.svg
    alt: "GaussianMarkovRandomFields.jl"
  actions:
    - theme: brand
      text: "Get Started"
      link: /tutorials/
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


## What are Gaussian Markov Random Fields? {#What-are-Gaussian-Markov-Random-Fields?}

Gaussian Markov Random Fields (GMRFs) are Gaussian distributions with sparse precision (inverse covariance) matrices. This sparsity structure makes them computationally efficient for large-scale problems while maintaining the expressiveness needed for complex spatial and temporal modeling.

## Quick Start {#Quick-Start}

Install the package:

```julia
using Pkg
Pkg.add("GaussianMarkovRandomFields")
using GaussianMarkovRandomFields
```

