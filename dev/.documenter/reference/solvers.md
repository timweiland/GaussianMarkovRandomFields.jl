
# Solvers {#Solvers}

Fundamentally, all interesting quantities of GMRFs (samples, marginal variances, posterior means, ...) must be computed through **sparse linear algebra**. GaussianMarkovRandomFields.jl uses [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl) as the backend for all linear algebra operations, providing access to a wide range of modern solvers and preconditioners while maintaining a unified interface. Our goal is to provide sane defaults that &quot;just work&quot; for most users, while still allowing power users to customize the behavior through LinearSolve.jl algorithms. Read further to learn about the available capabilities and how to use them.

## Direct Methods (Cholesky, LU) {#Direct-Methods-Cholesky,-LU}

[CHOLMOD](https://github.com/DrTimothyAldenDavis/SuiteSparse) and other direct solvers are  state-of-the-art methods for sparse linear systems. LinearSolve.jl automatically selects appropriate direct methods (typically Cholesky for symmetric positive definite systems like GMRF precision matrices) to compute factorizations of the precision matrix, which are then leveraged to draw samples and compute posterior means. Marginal variances are computed using selected inversion when available, or RBMC as a fallback.

## Pardiso {#Pardiso}

[Pardiso](https://panua.ch/pardiso/) is a state-of-the-art direct solver for sparse linear systems with excellent performance on large-dimensional GMRFs and highly efficient methods for marginal variance computation.

To use Pardiso, you need to set up and load [Pardiso.jl](https://github.com/JuliaSparse/Pardiso.jl), then specify `alg=LinearSolve.PardisoJL()` when creating your GMRF:

```julia
using Pardiso, LinearSolve
gmrf = GMRF(μ, Q, alg=LinearSolve.PardisoJL())
```


See the [LinearSolve.jl documentation](https://docs.sciml.ai/LinearSolve/stable/solvers/solvers/#LinearSolve.PardisoJL) for more information about PardisoJL algorithm options.

## Iterative Methods (CG, GMRES) {#Iterative-Methods-CG,-GMRES}

The Conjugate Gradient (CG) method and other iterative methods are efficient approaches for solving large sparse symmetric positive-definite linear systems. LinearSolve.jl provides access to various iterative solvers that can be used for large-scale GMRFs where direct methods become prohibitively expensive.

For symmetric systems like GMRFs, CG-based methods are typically most appropriate:

```julia
using LinearSolve
gmrf = GMRF(μ, Q, alg=LinearSolve.KrylovJL_CG())
```


## Variance Computation Strategies {#Variance-Computation-Strategies}

Fundamentally, computing marginal variances of GMRFs is not trivial, as it requires computing the diagonal entries of the covariance matrix (which is the inverse of the precision matrix).

**Selected Inversion** (including Takahashi recursions) is a highly accurate and stable method for computing these variances when available for the chosen algorithm. This is automatically used when supported.

When selected inversion is not available, the package automatically falls back to **RBMC** (Rao-Blackwellized Monte Carlo), a sampling-based approach that is less accurate but can be much faster for large GMRFs.
<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.AbstractVarianceStrategy' href='#GaussianMarkovRandomFields.AbstractVarianceStrategy'><span class="jlbinding">GaussianMarkovRandomFields.AbstractVarianceStrategy</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
AbstractVarianceStrategy
```


An abstract type for a strategy to compute the variance of a GMRF.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/typedefs.jl#L5-L9" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.RBMCStrategy' href='#GaussianMarkovRandomFields.RBMCStrategy'><span class="jlbinding">GaussianMarkovRandomFields.RBMCStrategy</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
RBMCStrategy(n_samples; rng)
```


Rao-Blackwellized Monte Carlo estimator of a GMRF&#39;s marginal variances based on [[3](/bibliography#Siden2018)]. Particularly useful in large-scale regimes where Takahashi recursions may be too expensive.

**Arguments**
- `n_samples::Int`: Number of samples to draw.
  

**Keyword arguments**
- `rng::Random.AbstractRNG = Random.default_rng()`: Random number generator.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/solvers/rbmc.jl#L8-L21" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.BlockRBMCStrategy' href='#GaussianMarkovRandomFields.BlockRBMCStrategy'><span class="jlbinding">GaussianMarkovRandomFields.BlockRBMCStrategy</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
BlockRBMCStrategy(n_samples; rng, enclosure_size)
```


Block Rao-Blackwellized Monte Carlo estimator of a GMRF&#39;s marginal variances based on [[3](/bibliography#Siden2018)]. Achieves faster convergence than plain RBMC by considering blocks of nodes rather than individual nodes, thus integrating more information about the precision matrix. `enclosure_size` specifies the size of these blocks. Larger values lead to faster convergence (in terms of the number of samples) at the cost of increased compute. Thus, one should aim for a sweet spot between sampling costs and block operation costs.

**Arguments**
- `n_samples::Int`: Number of samples to draw.
  

**Keyword arguments**
- `rng::Random.AbstractRNG = Random.default_rng()`: Random number generator.
  
- `enclosure_size::Int = 1`: Size of the blocks.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/solvers/rbmc.jl#L31-L51" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Advanced Operations {#Advanced-Operations}
<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.logdet_cov' href='#GaussianMarkovRandomFields.logdet_cov'><span class="jlbinding">GaussianMarkovRandomFields.logdet_cov</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
logdet_cov(linsolve, alg)
```


Compute the log determinant of the covariance matrix (inverse of precision). Dispatches on the algorithm type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/solvers/logdet.jl#L6-L11" target="_blank" rel="noreferrer">source</a></Badge>



```julia
logdet_cov(linsolve)
```


Convenience function that dispatches to logdet_cov(linsolve, linsolve.alg).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/solvers/logdet.jl#L17-L21" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.selinv' href='#GaussianMarkovRandomFields.selinv'><span class="jlbinding">GaussianMarkovRandomFields.selinv</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
selinv(linsolve, alg)
```


Compute the full selected inverse matrix using selected inversion. Dispatches on the algorithm type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/solvers/selinv.jl#L47-L52" target="_blank" rel="noreferrer">source</a></Badge>



```julia
selinv(linsolve)
```


Convenience function that dispatches to selinv(linsolve, linsolve.alg).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/solvers/selinv.jl#L58-L62" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.backward_solve' href='#GaussianMarkovRandomFields.backward_solve'><span class="jlbinding">GaussianMarkovRandomFields.backward_solve</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
backward_solve(linsolve, x, alg)
```


Perform backward solve L^T \ x where L is the Cholesky factor. Dispatches on the algorithm type.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/solvers/backward_solve.jl#L29-L34" target="_blank" rel="noreferrer">source</a></Badge>



```julia
backward_solve(linsolve, x)
```


Convenience function that dispatches to backward_solve(linsolve, x, linsolve.alg).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/solvers/backward_solve.jl#L40-L44" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Choosing a Solver {#Choosing-a-Solver}

LinearSolve.jl automatically selects appropriate algorithms based on matrix properties:
- **Direct methods** (Cholesky, LU) for small to medium-sized GMRFs
  
- **Iterative methods** (CG, GMRES) for large sparse GMRFs where direct methods become prohibitively expensive
  

This matches our general recommendations: Direct solvers combined with selected inversion are highly accurate and stable and should be used whenever possible. However, direct solvers become prohibitively expensive for very large-scale GMRFs, both in terms of compute and memory use. In these regimes, iterative methods may still be a viable option.

If you have access to both strong parallel computing resources and a Pardiso license, `LinearSolve.PardisoJL()` may provide the best performance.
