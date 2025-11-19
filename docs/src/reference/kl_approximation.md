# KL-minimizing Sparse GP Approximations

Sparse GMRF approximations to Gaussian processes defined by kernel (covariance) matrices using KL-divergence minimizing sparse Cholesky factorization.

These functions enable efficient large-scale GP inference by approximating a dense kernel matrix with a sparse precision matrix using spatially-informed Cholesky factorization that minimizes the Kullback-Leibler divergence.

## Overview

The KL-divergence minimizing sparse Cholesky approximation constructs a sparse GMRF from a kernel matrix through a four-stage algorithm that exploits spatial structure to determine which precision matrix entries can be safely set to zero.

### 1. Reverse Maximin Ordering

The algorithm begins by computing a reverse maximin ordering of the input points. This ordering selects points greedily to maximize the distance to previously selected points, creating a natural hierarchical structure where similar points tend to appear consecutively. The result is a permutation vector `P` that reorders the points for optimal sparsity.

### 2. Sparsity Pattern Construction

Based on the ordering and spatial locations, the algorithm determines a sparsity pattern using a neighborhood radius `ρ × ℓᵢ`, where `ℓᵢ` is the maximin distance for point `i`. This radius defines which matrix entries will be nonzero. Larger `ρ` values create denser (and more accurate) approximations at the cost of increased computational expense.

### 3. Supernodal Clustering (Optional, Default)

By default, the algorithm uses supernodal clustering to group columns with similar sparsity patterns into "supernodes". Rather than solving many small linear systems (one per column), the algorithm solves fewer but larger systems (one per supernode). This tends to be much more efficient since larger matrix operations can take advantage of optimized BLAS routines.

The clustering parameter `λ` (typically 1.5) controls the similarity threshold: columns are grouped together if their maximin distances differ by less than this factor.

### 4. Cholesky Factorization

The algorithm fills in the sparse Cholesky factor `L` such that `L * L' ≈ (PKP')⁻¹`, where `K` is the covariance matrix. For each supernode (or individual column when not using supernodes), the algorithm extracts the relevant submatrix of the covariance, solves a small dense Cholesky problem, and fills in the corresponding entries of `L`. The result is a GMRF with sparse precision matrix `Q ≈ K⁻¹` that enables efficient inference.

## Algorithm Parameters

The algorithm has two main tuning parameters:

The sparsity radius parameter `ρ` (default: 2.0) controls the neighborhood size for determining sparsity. Larger values create denser approximations with better accuracy but higher computational cost. Typical values range from 1.5 to 3.0, with 2.0-2.5 being a good balanced choice for most applications.

The supernodal clustering threshold `λ` (default: 1.5) controls how columns are grouped into supernodes. Setting it to `nothing` disables supernodal clustering entirely, which can be useful for very small problems or debugging. Higher values create larger supernodes, which can improve performance if the problem structure is favorable. 

## Main Functions

```@docs
approximate_gmrf_kl
sparse_approximate_cholesky
sparse_approximate_cholesky!
```

## Supporting Functions

### Point Ordering

```@docs
reverse_maximin_ordering
reverse_maximin_ordering_and_sparsity_pattern
```

### Supernodal Clustering

```@docs
SupernodeClustering
form_supernodes
```

The `form_supernodes` function takes the sparsity pattern and maximin ordering and creates a supernodal clustering. This is used internally when `λ !== nothing`.

### Matrix Utilities

```@docs
PermutedMatrix
```

The `PermutedMatrix` wrapper enables efficient access to permuted covariance matrices without materializing the full permuted matrix in memory.

## Performance Considerations

### When to Use Supernodal Factorization

The default supernodal factorization (`λ=1.5`) is recommended for most use cases, as it's typically faster and more accurate. You should only consider disabling it (`λ=nothing`) for very small problems (fewer than 100 points), extremely memory-constrained environments, or when debugging or comparing against a reference implementation.

## See Also

- [KL-minimizing Sparse GMRF Approximations to Gaussian Processes](@ref) tutorial demonstrates the full workflow with examples
- [Schaefer2021](@cite) for the full paper describing these algorithms in detail
