# Formula Interface

The formula interface lets you compose models succinctly using `StatsModels.jl` syntax.
It maps terms on the right-hand side to latent components and assembles a sparse
design matrix that connects observations to the combined latent field.
To use the formula syntax, load `StatsModels` first to activate the corresponding package extension.

## Quick Example

```julia
using GaussianMarkovRandomFields, StatsModels, Distributions, SparseArrays

# Suppose W is a spatial adjacency for regions; y are counts with an offset
besag = Besag(W)                         # structured spatial effect
f = @formula(y ~ 1 + x + IID(region) + besag(region))
comp = build_formula_components(f, data; family = Poisson)
lik  = comp.obs_model(data.y; offset = data.logE)
prior = comp.combined_model(; τ_besag=1.0, τ_iid=1.0)
post  = gaussian_approximation(prior, lik)
```

## Terms and Builders

```@docs
GaussianMarkovRandomFields.IID
GaussianMarkovRandomFields.RandomWalk
GaussianMarkovRandomFields.AR1
GaussianMarkovRandomFields.Besag
GaussianMarkovRandomFields.build_formula_components
```

## See Also

- The BYM + fixed effects Poisson tutorial shows this in practice: [Advanced GMRF modelling for disease mapping](@ref)

