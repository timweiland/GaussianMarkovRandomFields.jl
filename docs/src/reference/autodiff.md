# Automatic Differentiation Reference

GaussianMarkovRandomFields.jl provides automatic differentiation support for gradient-based inference and optimization. This page documents the capabilities and limitations.

## Supported AD Backends

We support the following AD backends:

- **Zygote.jl**: General-purpose, good first choice, works in most cases
- **Mooncake.jl**: Modern reverse-mode AD; requires the CliqueTrees linear solver (see below)
- **Enzyme.jl**: Often 2-5× faster than Zygote, but requires attention to type stability
- **ForwardDiff.jl**: Fast in the right situations (i.e. for functions with few inputs)

All backends produce mathematically identical gradients.

## Supported Operations

The package provides custom AD rules for the following operations:

1. **GMRF construction**: `GMRF(μ, Q, algorithm)` - gradients flow through mean μ and precision matrix Q
2. **Log-probability density**: `logpdf(gmrf, z)` - uses selected inversion for efficient gradient computation
3. **Gaussian approximation**: `gaussian_approximation(prior_gmrf, obs_lik)` - uses Implicit Function Theorem to avoid differentiating through the optimization loop

All three operations work with both regular `GMRF` and `ConstrainedGMRF` priors (e.g. from `RW1Model`, `BesagModel`). Constrained GMRF support is available for Zygote and ForwardDiff.

## Mooncake and the CliqueTrees Backend

Mooncake support is provided through the pure-Julia CliqueTrees solver backend.
Construct your GMRFs with `LinearSolve.CliqueTreesFactorization()` and load
`Mooncake` together with `MooncakeSparse`:

```julia
using GaussianMarkovRandomFields, LinearSolve
using DifferentiationInterface
using Mooncake, MooncakeSparse

function objective(θ)
    Q = build_precision(θ)
    prior = GMRF(build_mean(θ), Q, CliqueTreesFactorization())
    posterior = gaussian_approximation(prior, obs_lik)
    return logpdf(posterior, y)
end

grad = DifferentiationInterface.gradient(objective, AutoMooncake(), θ)
```

Supported operations under Mooncake: GMRF construction, `logpdf`, `var`
(marginal variances via selected inversion), and `gaussian_approximation`
(including the conjugate Normal paths). The same support is available for
[`ChordalGMRF`](@ref), which wraps the same factorization outside the
LinearSolve interface.

The workspace path is supported as well: construct the workspace with the
CliqueTrees backend and reuse it across gradient evaluations —

```julia
ws = GMRFWorkspace(Q_pattern, CliqueTreesBackend)

function objective(θ)
    prior = WorkspaceGMRF(build_mean(θ), build_precision(θ), ws)
    posterior = gaussian_approximation(prior, obs_lik)
    return logpdf(posterior, y)
end
```

Linear equality constraints (e.g. sum-to-zero) are fully supported under
Mooncake on both constrained paths — `ConstrainedGMRF` over a
CliqueTrees-backed `GMRF`, and constrained `WorkspaceGMRF`s: `logpdf`, `var`,
and `gaussian_approximation` all differentiate, including through
`mean(posterior)`.

## Linear Solver Type Stability

When using Enzyme, type stability is critical. The default two-argument GMRF constructor uses a runtime dispatch to select an appropriate linear solver based on the precision matrix type. This is not type-stable and can cause issues with Enzyme.

To avoid these problems, always explicitly specify a linear solver when using Enzyme:

```julia
# For general sparse matrices
gmrf = GMRF(μ, Q, LinearSolve.CHOLMODFactorization())

# For SymTridiagonal matrices (AR1, RW1)
gmrf = GMRF(μ, Q, LinearSolve.LDLtFactorization())

# For diagonal matrices (IID)
gmrf = GMRF(μ, Q, LinearSolve.DiagonalFactorization())
```

See the [Automatic Differentiation Tutorial](../tutorials/automatic_differentiation.md) for a complete example.

## Current Limitations

- **Enzyme + ConstrainedGMRF**: The Enzyme extension does not yet have custom rules for `ConstrainedGMRF`. Use Zygote or ForwardDiff for constrained models.
- **ForwardDiff Hessians**: Second-order derivatives (Hessians) via nested ForwardDiff Duals are not supported. For Hessians, consider using finite-difference-over-ForwardDiff-gradient (`FiniteDiff.finite_difference_jacobian` of `ForwardDiff.gradient`), which is numerically more stable than pure finite-difference Hessians.

If you encounter AD issues or need support for additional operations, please open an issue on GitHub.

## See Also

- [Automatic Differentiation for GMRF Hyperparameters](@ref): Practical examples and workflows
- [ChainRulesCore.jl docs](https://juliadiff.org/ChainRulesCore.jl/stable/)
- [Enzyme.jl docs](https://enzyme.mit.edu/julia/stable/)
