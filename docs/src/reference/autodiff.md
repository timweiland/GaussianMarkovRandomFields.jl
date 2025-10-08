# Automatic Differentiation Reference

GaussianMarkovRandomFields.jl provides automatic differentiation support for gradient-based inference and optimization. This page documents the capabilities and limitations.

## Supported AD Backends

We support the following AD backends:

- **Zygote.jl**: General-purpose, good first choice, works in most cases
- **Enzyme.jl**: Often 2-5× faster than Zygote, but requires attention to type stability
- **ForwardDiff.jl**: Fast in the right situations (i.e. for functions with few inputs), but support is limited. See below.

All backends produce mathematically identical gradients.

## Supported Operations

The package provides custom AD rules for the following operations:

1. **GMRF construction**: `GMRF(μ, Q, algorithm)` - gradients flow through mean μ and precision matrix Q
2. **Log-probability density**: `logpdf(gmrf, z)` - uses selected inversion for efficient gradient computation
3. **Gaussian approximation**: `gaussian_approximation(prior_gmrf, obs_lik)` - uses Implicit Function Theorem to avoid differentiating through the optimization loop. **NOTE: We do not have a custom rule for ForwardDiff for this yet. Help is appreciated!**

These three operations cover most common GMRF workflows. If you need AD support for other operations, please open an issue on GitHub.

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

- **ForwardDiff**: Currently only has custom rules for the constructor and `logpdf`. AD through a Gaussian approximation is going to fail.

If you encounter AD issues or need support for additional operations, please open an issue on GitHub.

## See Also

- [Automatic Differentiation for GMRF Hyperparameters](@ref): Practical examples and workflows
- [ChainRulesCore.jl docs](https://juliadiff.org/ChainRulesCore.jl/stable/)
- [Enzyme.jl docs](https://enzyme.mit.edu/julia/stable/)
