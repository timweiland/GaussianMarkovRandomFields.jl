# API Reference

Every exported type and function is documented here, grouped by what it is for.
If you are after a worked example rather than a signature, the
[tutorials](../tutorials/index.md) are the better place to start.

Most workflows only touch the first four groups. Solvers, preconditioners and
linear maps matter when you are tuning performance or extending the package, and
can be ignored until then.

## Core types

- [GMRFs](gmrfs.md) — the distribution types themselves, and the operations they
  support: means, marginal variances, sampling, conditioning.
- [Hard Constraints](hard_constraints.md) — `ConstrainedGMRF`, for models that
  must satisfy linear equality constraints exactly.

## Building models

- [Latent Models](latent_models.md) — the `LatentModel` interface and the models
  that ship with the package: AR, RW, IID, Besag, BYM2, Matérn, and their
  combinations.
- [Autoregressive Models](autoregressive.md) — autoregressive and conditional
  autoregressive constructors.
- [Formula Interface](formula.md) — composing models with `StatsModels.jl`
  formula syntax, which assembles the latent components and design matrix for
  you.
- [Spatial Utilities](spatial_utils.md) — building adjacency structures, such as
  contiguity matrices from polygons.

## SPDE discretizations

- [SPDEs](spdes.md) — the `SPDE` type and the equations that ship with the
  package.
- [Discretizations](discretizations.md) — turning an SPDE into a GMRF, in space
  and in space-time.
- [Meshes](meshes.md) — helpers for constructing the finite element meshes those
  discretizations need.

## Observations and inference

- [Observation Models](observation_models.md) — how data relates to the latent
  field, from exponential families to custom autodiff-defined likelihoods.
- [Gaussian Approximation](gaussian_approximation.md) — Laplace approximation of
  the posterior for non-Gaussian likelihoods.
- [Automatic Differentiation](autodiff.md) — supported backends, what is
  differentiable, and the current limitations.

## Computation and performance

- [Solvers](solvers.md) — the sparse linear algebra backends that everything
  else is built on.
- [Preconditioners](preconditioners.md) — preconditioning for iterative solvers.
- [Workspaces](workspaces.md) — reusing a symbolic factorization across many
  precision matrices with the same sparsity pattern.
- [Linear maps](linear_maps.md) — the structured matrix types used to keep
  precision matrices cheap.

## Further topics

- [KL Approximations](kl_approximation.md) — approximating a kernel-defined
  Gaussian process by a sparse GMRF, without going via an SPDE.
- [Graphical Lasso](graphical_lasso.md) — estimating a sparse precision matrix
  from data when the structure is unknown.
- [Plotting](plotting.md) — Makie recipes for spatial and spatiotemporal GMRFs.
