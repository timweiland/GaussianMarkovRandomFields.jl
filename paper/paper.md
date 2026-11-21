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
The package includes various methods to construct GMRFs, including the stochastic partial differential equation (SPDE) approach [@lindgrenExplicitLinkGaussian2011], and provides efficient, customizable routines for GMRF computations.
It is designed to be intuitive to use for rapid prototyping, yet sufficiently flexible to empower expert users to solve advanced problems.
As such, it is suitable for both research and teaching in spatial statistics and Bayesian modeling.

# Statement of Need

GMRFs have become a cornerstone of modern spatial statistics, yet practical barriers limit their accessibility.
The priors of interest are rarely specified directly through a sparse precision matrix: a Matérn field on an irregular domain, for instance, only becomes a GMRF once a stochastic partial differential equation has been discretized with finite elements, and intrinsic models such as random walks additionally need linear constraints.
Implementing GMRF-based models requires navigating sparse linear algebra libraries and finite element discretizations, and choosing a linear solver suited to the problem (a sparse Cholesky factorization, an LDL$^\top$ factorization for tridiagonal precisions, or a preconditioned iterative method), all of which create a steep barrier to entry for domain scientists.

`GaussianMarkovRandomFields.jl` eliminates these barriers by providing a feature-rich toolkit for creation and manipulation of GMRFs.
Once constructed, a GMRF can be conditioned on Gaussian observations in closed form, approximated by a Gaussian under any other likelihood, constrained, sampled, and queried for log-densities and marginal variances, with gradients of all of these with respect to its hyperparameters.
These operations are the building blocks of latent Gaussian inference schemes such as INLA and TMB (see the Research Impact Statement).
The package offers utilities to create well-known latent models, ranging from simple autoregressive and Besag [@besagConditionalIntrinsicAutoregressions1995] models to more involved spatial and spatiotemporal finite element-based models [@lindgrenExplicitLinkGaussian2011; @clarottoSPDEApproachSpatiotemporal2024].
For users requiring continuous spatial domains, the SPDE approach is implemented through integration with Ferrite.jl [@carlsson2025ferrite], enabling principled approximations of Matérn and related Gaussian processes on arbitrary geometries.
As an alternative to SPDE-based constructions, the package also implements KL-minimizing sparse Cholesky approximations [@schaeferSparseCholeskyKL2021], which construct sparse precision matrices directly from covariance specifications and are particularly effective for covariance functions with strong screening effects.

The package extends beyond Gaussian observations through Gaussian approximation methods.
When working with non-Gaussian data such as counts or binary outcomes, the package seamlessly handles exponential family likelihoods.
Users may also specify custom likelihood functions, for which the package exploits sparsity structure through automatic differentiation (AD) to maintain computational efficiency even for complex observation models, an approach established by Skaug and Fournier [@skaug2006automatic] and popularised by TMB [@kristensen2016tmb].
For large-scale applications, the package provides efficient marginal variance computation methods [@lin2011selinv; @rueMarginalVariancesGaussian2005a; @sidenEfficientCovarianceApproximations2018a].
Because the posterior is Gaussian, or approximated by a Gaussian, its marginals are fully described by the mean and the marginal variances, which is all that marginal inference requires.
\autoref{fig:bernoulli} demonstrates both capabilities on a spatial binary classification task.
AD support via ForwardDiff.jl [@revels2016forward], Enzyme.jl [@moses2020enzyme], Zygote.jl [@innes2018zygote], Mooncake.jl [@tebbutt2024mooncake] and DifferentiationInterface.jl [@dalle2025differentiationinterface] enables gradient-based optimization and inference methods, making the package compatible with modern probabilistic programming frameworks including Turing.jl [@ge2018turing].

![Spatial binary classification of tree species in the Lansing Woods dataset [@gerrard1969lansing; @baddeley2015spatstat] using a Matérn latent field with Bernoulli observations. Left: predicted probability of hickory, from a Gaussian approximation of the posterior; points show observed trees: \textcolor{red}{hickory (circles)} and \textcolor{blue}{other species (diamonds)}. Right: posterior marginal standard deviation of the latent field, computed by selected inversion. It is lowest where trees are dense and grows towards the edges of the domain, a known boundary effect of SPDE models.\label{fig:bernoulli}](bernoulli_classification.pdf){ width=100% }

# State of the Field

Implementations of GMRF-based inference have historically centered on R.
R-INLA [@rueApproximateBayesianInference2009] provides comprehensive support for latent Gaussian models through integrated nested Laplace approximations, but its model library is curated, and custom extensions require low-level C implementation.
The inlabru package [@bachl2019inlabru] adds a flexible modeling interface for non-linear predictors while remaining tied to R-INLA's computational backend.
The rSPDE package [@bolin2025rspde] supports fractional SPDE models with non-integer smoothness.
Template Model Builder (TMB) [@kristensen2016tmb] supports Laplace approximation and gradient-based inference over latent Gaussian models via C++ templates, but requires users to write C++ for their models.
In Python, general-purpose probabilistic programming frameworks such as PyMC [@abril2023pymc] expose some GMRF primitives but lack specialized machinery.
Such a framework can represent a GMRF as a multivariate normal distribution parameterized by its precision, but its default inference samples the latent field jointly with the hyperparameters.
For large spatial fields this scales poorly, since the posterior is high-dimensional and strongly correlated; the efficient alternative integrates the field out with a Laplace approximation, which needs sparse factorizations, selected inversion and the approximation itself, the machinery this package provides.

Across these packages, GMRFs appear as implementation details of a particular inference algorithm, buried in C or C++ code and not exposed as first-class objects that users can extend.
`GaussianMarkovRandomFields.jl` instead treats the GMRF as the primary abstraction, exposed through a high-level interface, which enables workflows that are impractical elsewhere.

The package is designed to be extended rather than to offer a closed catalogue of models: neither the observation likelihood nor the latent prior need come from a fixed list, as both may be ordinary Julia functions, and the sparsity of their Hessians is discovered automatically.
A latent field whose own dynamics are nonlinear (e.g., a state-space model of the kind routinely written in C++ for TMB [@nielsen2014sam]) is therefore expressed directly, and the iterated Laplace approximation such a model requires follows from the same machinery.
New latent models, GMRF types and solver backends plug in by implementing small interfaces, and then work with the existing inference routines.
Because a GMRF is itself a distribution with automatic differentiation rules attached, it composes directly with probabilistic programming languages: gradient-based samplers such as NUTS [@hoffman2014nuts] target the latent field and its hyperparameters jointly, giving fully Bayesian inference [@ge2018turing].
SPDE discretizations are delegated to a general finite element library [@carlsson2025ferrite], putting custom differential operators, meshes, and boundary conditions within reach; the capability underlying [@weiland2025gmrfpde].

# Software Design

The package is organized around an abstract type hierarchy centered on `AbstractGMRF`.
Concrete subtypes, including the default `GMRF`, the hard-constrained `ConstrainedGMRF`, and the model-wrapping `MetaGMRF`, expose a uniform interface for mean, precision, `logpdf`, sampling, and marginal variance, so downstream code is written once and works across variants.

A separate `LatentModel` abstraction decouples model specification (hyperparameters mapped to sparse precision) from construction of GMRF instances.
Provided models include autoregressive processes AR(p), random walks, IID, fixed effects, Besag, BYM2, Matérn SPDE, and compositions thereof; users add their own by implementing a small protocol.
A formula interface built on StatsModels.jl offers the familiar R-style syntax for combined models.

The following lines fit the model of \autoref{fig:bernoulli}, given point coordinates `X` and binary labels `y`.
Both the prior and the approximate posterior are ordinary GMRFs, so everything that works on one works on the other:

```julia
using GaussianMarkovRandomFields, Distributions
using Ferrite, FerriteGmsh, Gmsh, LibGEOS  # activate the FEM extension

latent = MaternModel(X; smoothness = 1)  # SPDE prior on a mesh around X
prior = latent(τ = 1.0, range = 0.2)     # a GMRF
obs_model = PointEvaluationObsModel(latent.discretization, X, Bernoulli)
post = gaussian_approximation(prior, obs_model(y))  # again a GMRF
μ, σ = mean(post), std(post)             # σ via selected inversion
```

Linear solves are delegated to LinearSolve.jl [@kimmerer2026linearsolve], so users can swap CHOLMOD, Pardiso, LDL$^\top$ factorizations, or preconditioned conjugate gradient without touching model code.
AD is provided through package extensions for ForwardDiff.jl, Zygote.jl/ChainRulesCore.jl, Enzyme.jl and Mooncake.jl, with chain rules written at the GMRF-construction level so gradients flow through composed models.
This split (abstract core, composable solver backends, AD via extensions) keeps optional dependencies optional while preserving extensibility.

To assess performance, we timed a Gaussian approximation with all latent marginal variances for a Poisson model with a Matérn SPDE prior at fixed hyperparameters against R-INLA [@rueApproximateBayesianInference2009], on identical meshes and data and single-threaded.
The two posteriors agree to within $10^{-7}$.
Measured against R-INLA's own timing of its inference, which excludes its R-side preparation and solver start-up, `GaussianMarkovRandomFields.jl` was 3.4 to 4.2 times faster for latent dimensions from 529 to 57,030 (1.8 s versus 5.9 s at the largest), with R-INLA using its default TAUCS solver.
The benchmark scripts are in the repository's `paper/benchmark` directory.

# Research Impact Statement

The package has supported methodological research on probabilistic PDE solvers [@weiland2025gmrfpde].
Near-term impact is rooted in the package's scope: `GaussianMarkovRandomFields.jl` supplies the foundational infrastructure needed to bring well-established latent-Gaussian inference methodologies to the Julia ecosystem.

Integrated Nested Laplace Approximation (INLA) [@rueApproximateBayesianInference2009] and Template Model Builder (TMB) [@kristensen2016tmb], with HMC extensions such as tmbstan [@monnahan2018tmbstan], are widely used in ecology, epidemiology, and environmental statistics, but have been available only in R or as C++ templates.
The sparse-linear-algebra, automatic-differentiation, and latent-model abstractions provided here are the prerequisite machinery for Julia implementations of these methods.
`Latte.jl` [@weiland2026latte], a probabilistic programming language for latent Gaussian models built by the author on these abstractions, provides INLA, TMB-style Laplace approximation, and HMC-Laplace behind a model-specification syntax --- evidence that the machinery described here is sufficient for that purpose, not merely intended for it.
Integration with Turing.jl [@ge2018turing] additionally enables full MCMC over GMRF hyperparameters, combining GMRF sub-models with arbitrary non-Gaussian components in a probabilistic program.

The package targets researchers and practitioners working in spatial statistics, epidemiology, environmental science, and related fields, and can serve both as a turnkey toolkit and as a substrate for methodological development in Julia's ecosystem for AD and scientific machine learning.

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
