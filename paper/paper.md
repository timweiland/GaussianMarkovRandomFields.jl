---
title: 'GaussianMarkovRandomFields.jl: Flexible Latent Gaussian Modeling with Sparse Precision in Julia'
                                       
tags:
  - Julia
  - Bayesian inference
  - Sparse linear algebra
  - Spatial statistics
authors:
  - name: Tim Weiland
    orcid: 0009-0003-7636-5970
    affiliation: 1
affiliations:
 - name: University of Tübingen, Germany
   index: 1
date: 29 December 2025
bibliography: paper.bib
---

# Summary

Bayesian inference provides a strong theoretical foundation to draw conclusions from real-world data, with applications across all of the sciences.
Yet in practice, algorithms for Bayesian inference often require excessive computational resources, limiting their applicability to large-scale problems.
Particularly in the context of spatial and spatiotemporal settings, Gaussian Markov Random Fields (GMRFs) [@rueGaussianMarkovRandom2005a] offer a remedy to this issue.
GMRFs are Gaussian distributions with sparse precision matrices.
Inference under these models scales particularly favorably due to the use of sparse linear algebra routines.
This computational efficiency has led to their widespread adoption across various statistical applications [@lindgrenSPDEApproachGaussian2022].

`GaussianMarkovRandomFields.jl` provides a flexible and efficient Julia implementation of GMRFs.
The package includes various methods to construct GMRFs, including the SPDE approach [@lindgrenExplicitLinkGaussian2011], and provides efficient, customizable routines for GMRF computations.
It is designed to be intuitive to use for rapid prototyping, yet sufficiently flexible to empower expert users to solve advanced problems.
As such, it is suitable for both research and teaching in spatial statistics and Bayesian modeling.

# Statement of Need

GMRFs have become a cornerstone of modern spatial statistics, yet practical barriers limit their accessibility.
Implementing GMRF-based models requires navigating sparse linear algebra libraries, finite element discretizations, and problem-specific solver selection, all of which create a steep barrier to entry for domain scientists.

`GaussianMarkovRandomFields.jl` eliminates these barriers by providing a feature-rich toolkit for creation and manipulation of GMRFs.
The package offers utilities to create well-known latent models, ranging from simple autoregressive and Besag [@besagConditionalIntrinsicAutoregressions1995] models to more involved spatial and spatiotemporal finite element-based models [@lindgrenExplicitLinkGaussian2011; @clarottoSPDEApproachSpatiotemporal2024].
For users requiring continuous spatial domains, the SPDE approach is implemented through integration with Ferrite.jl [@carlsson2025ferrite], enabling principled approximations of Matérn and related Gaussian processes on arbitrary geometries.
As an alternative to SPDE-based constructions, the package also implements KL-minimizing sparse Cholesky approximations [@schaeferSparseCholeskyKL2021], which construct sparse precision matrices directly from covariance specifications and are particularly effective for covariance functions with strong screening effects.

The package extends beyond Gaussian observations through Gaussian approximation methods.
When working with non-Gaussian data such as counts or binary outcomes, the package seamlessly handles exponential family likelihoods and custom likelihood functions, exploiting sparsity structure through automatic differentiation to maintain computational efficiency even for complex observation models.
For large-scale applications, the package provides efficient marginal variance computation methods [@lin2011selinv; @rueMarginalVariancesGaussian2005a; @sidenEfficientCovarianceApproximations2018a].
\autoref{fig:bernoulli} demonstrates this capability on a spatial binary classification task.
Automatic differentiation support via ForwardDiff.jl [@revels2016forward] and Enzyme.jl [@moses2020enzyme] enables gradient-based optimization and inference methods, making the package compatible with modern probabilistic programming frameworks including Turing.jl [@ge2018turing].

![Spatial binary classification of tree species in the Lansing Woods dataset using a Matérn latent field with Bernoulli observations. Points show observed trees: \textcolor{red}{hickory} and \textcolor{blue}{other species}. The heatmap shows predicted probabilities, demonstrating the package's support for non-Gaussian likelihoods through efficient Gaussian approximation.\label{fig:bernoulli}](bernoulli_classification.pdf){ width=80% }

# State of the Field

Implementations of GMRF-based inference have historically centered on R.
R-INLA [@rueApproximateBayesianInference2009] provides comprehensive support for latent Gaussian models through integrated nested Laplace approximations, but its model library is curated and custom extensions require low-level C implementation.
The inlabru package [@bachl2019inlabru] adds a flexible modeling interface for non-linear predictors while remaining tied to R-INLA's computational backend.
The rSPDE package [@bolin2025rspde] supports fractional SPDE models with non-integer smoothness.
Template Model Builder (TMB) [@kristensen2016tmb] supports Laplace approximation and gradient-based inference over latent Gaussian models via C++ templates, but requires users to write C++ for their models.
In Python, general-purpose probabilistic programming frameworks such as PyMC [@abril2023pymc] expose some GMRF primitives but lack specialized machinery.

Across these packages, GMRFs appear as implementation details of a specific inference algorithm, buried in C or C++ code and not exposed as first-class objects that users can extend.
`GaussianMarkovRandomFields.jl` instead treats GMRFs as the primary abstraction and exposes a high-level interface that is straightforward for practitioners to use, extend, and compose with existing tools.
It combines the computational efficiency that sparse-precision Gaussian inference demands with the flexibility required for methodological research.
The design leverages Julia's multiple dispatch to integrate directly with LinearSolve.jl, Ferrite.jl, ForwardDiff.jl [@revels2016forward], Enzyme.jl [@moses2020enzyme], and the broader SciML stack, a combination that no existing GMRF package offers.

# Software Design

The package is organized around an abstract type hierarchy centered on `AbstractGMRF`.
Concrete subtypes, including the default `GMRF`, the hard-constrained `ConstrainedGMRF`, and the model-wrapping `MetaGMRF`, expose a uniform interface for mean, precision, `logpdf`, sampling, and marginal variance, so downstream code is written once and works across variants.

A separate `LatentModel` abstraction decouples model specification (hyperparameters mapped to sparse precision) from construction of GMRF instances.
Provided models include autoregressive processes AR(p), random walks, IID, fixed effects, Besag, BYM2, Matérn SPDE, and compositions thereof; users add their own by implementing a small protocol.
A formula interface built on StatsModels.jl offers the familiar R-style syntax for combined models.

Linear solves are delegated to LinearSolve.jl, so users can swap CHOLMOD, Pardiso, LDLt factorizations, or preconditioned conjugate gradient without touching model code.
Automatic differentiation is provided through package extensions for ForwardDiff.jl, Zygote.jl/ChainRulesCore.jl, and Enzyme.jl, with chain rules written at the GMRF-construction level so gradients flow through composed models.
This split (abstract core, composable solver backends, AD via extensions) keeps the package's load-time cost small while preserving extensibility.

# Research Impact Statement

The package has supported methodological research on probabilistic PDE solvers [@weiland2025gmrfpde].
Near-term impact is rooted in the package's scope: `GaussianMarkovRandomFields.jl` supplies the foundational infrastructure needed to bring well-established latent-Gaussian inference methodologies to the Julia ecosystem.

Integrated Nested Laplace Approximation (INLA) [@rueApproximateBayesianInference2009] and Template Model Builder (TMB) [@kristensen2016tmb], with HMC extensions such as tmbstan [@monnahan2018tmbstan], are widely used in ecology, epidemiology, and environmental statistics, but have been available only in R or as C++ templates.
The sparse-linear-algebra, automatic-differentiation, and latent-model abstractions provided here are the prerequisite machinery for Julia implementations of these methods.
Integration with Turing.jl [@ge2018turing] additionally enables full MCMC over GMRF hyperparameters, combining GMRF sub-models with arbitrary non-Gaussian components in a probabilistic program.

The package targets researchers and practitioners working in spatial statistics, epidemiology, environmental science, and related fields, and can serve both as a turnkey toolkit and as a substrate for methodological development in Julia's AD- and SciML-aware ecosystem.

# AI Usage Disclosure

Claude Code (Anthropic's CLI coding assistant) was used as a development aid during the late stages of implementation, testing, and documentation, and when drafting portions of this paper.
All design decisions, mathematical specifications, and package scope are the author's own.
AI assistance was used to accelerate mechanical tasks (code translation, docstring drafting, routine refactors, prose tightening) under the author's direction.
Every AI-produced suggestion was reviewed, edited, and integrated by the author before commit or inclusion.
Correctness was verified through the package's test suite, which cross-checks implementations against dense-matrix baselines, finite-difference gradients, and analytical solutions where available, and by running the accompanying documentation examples.
No other generative-AI tools were used.

# Acknowledgements

The author gratefully acknowledges co-funding by the European Union (ERC, ANUBIS, 101123955).
Views and opinions expressed are however those of the author only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them.
The author also gratefully acknowledges the German Federal Ministry of Education and Research (BMBF) through the Tübingen AI Center (FKZ:01IS18039A); and funds from the Ministry of Science, Research and Arts of the State of Baden-Württemberg.
The author further thanks the International Max Planck Research School for Intelligent Systems (IMPRS-IS) for their support.

# References
