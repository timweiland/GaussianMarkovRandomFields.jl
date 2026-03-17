# Graphical Lasso

The graphical lasso learns a GMRF from data by estimating a sparse precision matrix.
Given a data matrix $X$, the algorithm solves

```math
\max_{\Omega \succ 0} \; \log\det \Omega - \operatorname{tr}(S\Omega) - \lambda \lVert \Omega \rVert_1
```

where $S$ is the sample covariance and $\lambda$ is a sparsity-inducing penalty.

The implementation follows the approach of [Zhang2018](@cite),
which combines soft-thresholding of the sample covariance with a maximum-determinant
positive-definite matrix completion via chordal graph techniques (using
[CliqueTrees.jl](https://github.com/AlgebraicJulia/CliqueTrees.jl)).

## Restricted Graphical Lasso

Instead of a scalar threshold $\lambda$, you can pass a sparse matrix whose sparsity
pattern defines which entries to penalize, and whose values define per-entry thresholds.
This is useful when the conditional independence structure is partially known.

## API Reference

```@docs
graphical_lasso
```
