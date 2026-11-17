# How computations work

A GMRF is defined by its mean $\mu$ and its sparse precision matrix $Q$. Every quantity
the package computes reduces to a handful of operations with $Q$. With a direct solver,
which is the default, they all build on a sparse Cholesky factorization $Q = L L^\top$
(after a fill-reducing reordering). It is computed once per precision matrix and shared by
all operations on it. Iterative solvers such as conjugate gradients never factorize $Q$:
they solve $Q x = b$ using only products with $Q$, and so support fewer of the operations
below. This page explains which operation each quantity needs, and which solvers provide
it.

## The operations

**Solves.** Conditioning on observations, and every Newton step of
[`gaussian_approximation`](@ref), needs solutions of $Q x = b$. With the factorization in
hand, that is one forward and one backward triangular solve with $L$.

**Log-determinants.** `logpdf`
needs $\log\det Q$, which is free once $Q$ is factorized:
$\log\det Q = 2 \sum_i \log L_{ii}$. The rest of the log-density is a sparse quadratic form.
[`logdet_cov`](@ref) exposes the log-determinant of the covariance $Q^{-1}$ directly.

**Backward solves, for sampling.** To draw $x \sim \mathcal{N}(\mu, Q^{-1})$, draw
$z \sim \mathcal{N}(0, I)$ and solve only the backward half, $L^\top v = z$, then set
$x = \mu + v$. This works because $\operatorname{Cov}(L^{-\top} z) = L^{-\top} L^{-1} = Q^{-1}$.
[`backward_solve`](@ref) performs this step.

**Selected inversion, for marginal variances.** `var` and `std` need the diagonal of
$Q^{-1}$. The full inverse is dense, but selected inversion (the Takahashi recursions)
computes $Q^{-1}$ only at the positions where $L$ is nonzero. Those include the diagonal,
and the cost is comparable to the factorization itself. See [`selinv`](@ref).

**Selected inversion, for gradients.** The derivative of the log-determinant is
$\partial_\theta \log\det Q = \operatorname{tr}(Q^{-1} \, \partial_\theta Q)$. Since
$\partial_\theta Q$ has the sparsity pattern of $Q$, only the entries of $Q^{-1}$ on that
pattern are needed, and they are a subset of what selected inversion computes. This is
why gradients of `logpdf`
and `logdetcov` with respect to the precision require a solver that supports selected
inversion; see [Automatic Differentiation](@ref "Automatic Differentiation Reference").

**RBMC, when selected inversion is not available.** Rao-Blackwellized Monte Carlo
estimates marginal variances from samples, using the law of total variance:

```math
\operatorname{Var}(x_i) = \frac{1}{Q_{ii}} + \operatorname{Var}\big(\mathbb{E}[x_i \mid x_{-i}]\big),
\qquad
\mathbb{E}[x_i \mid x_{-i}] = \mu_i - \frac{1}{Q_{ii}} \sum_{j \neq i} Q_{ij} (x_j - \mu_j).
```

The first term is exact; only the second is estimated from samples, which makes the
estimate much less noisy than the plain sample variance. [`RBMCStrategy`](@ref) and
[`BlockRBMCStrategy`](@ref) control the number of samples. The result is stochastic, so
prefer selected inversion whenever the solver supports it.

## Which solvers support what

| Algorithm | Solve | Log-determinant | Sampling | Marginal variances | Precision gradients |
|---|---|---|---|---|---|
| `CHOLMODFactorization` (default for sparse $Q$) | ✓ | ✓ | backward solve | selected inversion | ✓ |
| `CholeskyFactorization` (dense $Q$) | ✓ | ✓ | backward solve | selected inversion | ✓ |
| `LDLtFactorization` (`SymTridiagonal` $Q$) | ✓ | ✓ | backward solve | selected inversion | ✓ |
| `DiagonalFactorization` | ✓ | ✓ | backward solve | selected inversion | ✓ |
| `PardisoJL` (needs Pardiso.jl) | ✓ | ✓ | backward solve | selected inversion | ✓ |
| Iterative, e.g. `KrylovJL_CG` | ✓ | – | only with a supplied `Q_sqrt` | RBMC | – |

Without a factorization there is no $L$, so iterative solvers can sample only when a square
root of $Q$ is supplied: they solve $Q x = \sqrt{Q} z$, which has the right covariance. See
[Solvers](@ref) for how to choose an algorithm.

Two GMRF types use their own factorization instead of a LinearSolve algorithm, and support
all of the operations above:

- [`ChordalGMRF`](@ref) factorizes with the multifrontal Cholesky from CliqueTrees.jl,
  including selected inversion.
- [`WorkspaceGMRF`](@ref) factorizes through a reusable workspace, so that repeated factorizations
  of the same sparsity pattern share one symbolic analysis. See [Workspaces](@ref).
