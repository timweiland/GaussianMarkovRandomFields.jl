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
Implementing GMRF-based models requires navigating sparse linear algebra libraries, finite element discretizations, and problem-specific solver selection—all of which create a steep barrier to entry for domain scientists.

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

Existing GMRF software faces trade-offs between ease of use and flexibility.
R-INLA [@rueApproximateBayesianInference2009] provides comprehensive functionality for latent Gaussian models but is limited to pre-defined model structures and lacks flexibility for custom applications.
The inlabru package [@bachl2019inlabru] extends R-INLA with support for non-linear predictors and a more flexible interface, but remains tied to INLA's underlying computational framework.
The rSPDE package [@bolin2025rspde] enables fractional SPDE models with non-integer smoothness parameters.
General-purpose probabilistic programming frameworks like PyMC [@abril2023pymc] provide some GMRF functionality, but lack the specialized treatment of dedicated GMRF packages.
`GaussianMarkovRandomFields.jl` bridges this gap by combining ease of use with full extensibility, allowing users to leverage pre-built components while maintaining the flexibility to implement custom models and solvers.

The package is designed for researchers and practitioners working in spatial statistics, epidemiology, environmental science, and related fields requiring efficient Bayesian inference for spatially or temporally correlated data.
It serves both as a research tool for developing novel GMRF-based methods and as a teaching resource for courses in spatial statistics and Bayesian modeling.
The package has been used in methodological research on probabilistic PDE solvers [@weiland2025gmrfpde].

# Acknowledgements

The author gratefully acknowledges co-funding by the European Union (ERC, ANUBIS, 101123955).
Views and opinions expressed are however those of the author only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them.
The author also gratefully acknowledges the German Federal Ministry of Education and Research (BMBF) through the Tübingen AI Center (FKZ:01IS18039A); and funds from the Ministry of Science, Research and Arts of the State of Baden-Württemberg.
The author further thanks the International Max Planck Research School for Intelligent Systems (IMPRS-IS) for their support.

# References
