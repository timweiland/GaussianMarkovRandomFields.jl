
# Hard Constraints {#Hard-Constraints}

Many applications require probabilistic models to satisfy hard equality constraints. For instance, in spatial statistics we might need mass conservation (sum of values equals zero), or in engineering applications we might require boundary conditions to be exactly satisfied. The `ConstrainedGMRF` type enables efficient Bayesian inference under linear equality constraints while preserving the computational advantages of sparse precision matrices.

A `ConstrainedGMRF` represents the conditional distribution x | Ax = e, where x follows an unconstrained GMRF prior and A, e define the linear constraints. Despite the resulting distribution being degenerate (the precision matrix becomes singular), all essential operations—sampling, mean computation, variance calculation, and further conditioning—remain computationally efficient through conditioning by Kriging.

The key insight is that while the constrained distribution has infinite precision in the constraint directions, it retains finite variance in the orthogonal complement, enabling meaningful inference.

## Mathematical Foundation {#Mathematical-Foundation}

Given an unconstrained GMRF x ~ N(μ, Q⁻¹) and linear constraints Ax = e, the constrained distribution has:

**Constrained mean:**

```
μ_c = μ - Q⁻¹A^T(AQ⁻¹A^T)⁻¹(Aμ - e)
```


**Constrained covariance:**

```
Σ_c = Q⁻¹ - Q⁻¹A^T(AQ⁻¹A^T)⁻¹AQ⁻¹
```


**Sampling via Kriging:**

```
x_c = x - Q⁻¹A^T(AQ⁻¹A^T)⁻¹(Ax - e)
```


where x is drawn from the unconstrained distribution.

## Basic Usage Pattern {#Basic-Usage-Pattern}

```julia
# Step 1: Set up unconstrained prior GMRF
prior_gmrf = GMRF(μ_prior, Q_prior)

# Step 2: Define linear constraints Ax = e
A = constraint_matrix  # m × n matrix
e = constraint_vector  # m-dimensional vector

# Step 3: Create constrained GMRF
constrained_gmrf = ConstrainedGMRF(prior_gmrf, A, e)

# Step 4: Use like any other GMRF
constrained_sample = rand(constrained_gmrf)
constrained_mean = mean(constrained_gmrf)
marginal_vars = var(constrained_gmrf)
```


## Examples {#Examples}

### Sum-to-Zero Constraint {#Sum-to-Zero-Constraint}

A common constraint in spatial modeling ensures mass conservation:

```julia
using GaussianMarkovRandomFields, LinearAlgebra, SparseArrays

# Prior: independent components
n = 5
prior_gmrf = GMRF(ones(n), spdiagm(0 => ones(n)))

# Constraint: sum equals zero
A = ones(1, n)  # 1×5 matrix [1 1 1 1 1]
e = [0.0]       # sum = 0

# Constrained GMRF
constrained_gmrf = ConstrainedGMRF(prior_gmrf, A, e)

# Verify constraint satisfaction
x = rand(constrained_gmrf)
@assert abs(sum(x)) < 1e-10  # Constraint satisfied to numerical precision

# Linear conditioning also works seamlessly
y_obs = [0.3, -0.1]  # Observe first two components
A_obs = sparse([1.0 0.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0 0.0])
conditioned = linear_condition(constrained_gmrf; A=A_obs, Q_ϵ=10.0, y=y_obs)
```


### Multiple Constraints {#Multiple-Constraints}

Complex applications often require multiple simultaneous constraints:

```julia
# System with 4 variables
n = 4
prior_gmrf = GMRF(zeros(n), spdiagm(0 => ones(n)))

# Two constraints:
# 1. Sum equals zero: x₁ + x₂ + x₃ + x₄ = 0  
# 2. Symmetry: x₁ = x₂
A = [1.0 1.0 1.0 1.0;    # Sum constraint
     1.0 -1.0 0.0 0.0]   # Symmetry constraint
e = [0.0, 0.0]

constrained_gmrf = ConstrainedGMRF(prior_gmrf, A, e)

# Both constraints are automatically satisfied
x = rand(constrained_gmrf)
@assert abs(sum(x)) < 1e-10           # Sum = 0
@assert abs(x[1] - x[2]) < 1e-10      # x₁ = x₂
```


### Boundary Conditions in Spatial Models {#Boundary-Conditions-in-Spatial-Models}

In PDE-based spatial models, boundary conditions are naturally expressed as constraints:

```julia
# Spatial grid with fixed boundary values
n_interior = 16  # Interior points
n_boundary = 8   # Boundary points  
n_total = n_interior + n_boundary

# Prior on interior + boundary points
prior_gmrf = MaternSPDE(mesh, ν=1.5, κ=1.0)  # Some spatial prior

# Constraint: fix boundary values
A = [zeros(n_boundary, n_interior) I(n_boundary)]  # Select boundary points
e = boundary_values  # Known boundary conditions

constrained_gmrf = ConstrainedGMRF(prior_gmrf, A, e)
```


Constraints are also commonly used for identifiability in hierarchical models (e.g., sum-to-zero constraints on random effects in INLA-style modeling).

## Gaussian Approximation with Constraints {#Gaussian-Approximation-with-Constraints}

The real power emerges when combining constrained priors with non-Gaussian observation models. For non-conjugate cases, the `gaussian_approximation` function uses Fisher scoring while automatically respecting constraints throughout optimization:

```julia
# Constrained prior for log-rates (sum-to-zero for identifiability)
n = 6
prior_gmrf = GMRF(zeros(n), spdiagm(0 => ones(n)))
A = ones(1, n)
e = [0.0]
constrained_prior = ConstrainedGMRF(prior_gmrf, A, e)

# Poisson observations
obs_model = ExponentialFamily(Poisson)
y_counts = [5, 2, 8, 3, 6, 1]
obs_lik = obs_model(y_counts)

# Constrained Gaussian approximation to posterior
constrained_posterior = gaussian_approximation(constrained_prior, obs_lik)

# All samples from posterior satisfy the constraint
posterior_sample = rand(constrained_posterior)
@assert abs(sum(posterior_sample)) < 1e-10
```


For conjugate cases (Normal observations), specialized dispatches automatically use exact `linear_condition` instead of iterative optimization.

## API Reference {#API-Reference}
<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.ConstrainedGMRF' href='#GaussianMarkovRandomFields.ConstrainedGMRF'><span class="jlbinding">GaussianMarkovRandomFields.ConstrainedGMRF</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
ConstrainedGMRF{T,L,G} <: AbstractGMRF{T,L}
```


A Gaussian Markov Random Field with hard linear equality constraints.

Given an unconstrained GMRF x ~ N(μ, Q⁻¹) and constraints Ax = e, this represents the constrained distribution x | Ax = e.

This is a degenerate distribution (the precision matrix becomes singular), but sampling and mean computation are handled efficiently using conditioning by Kriging.

**Mathematical Background**

For x ~ N(μ, Q⁻¹) with constraint Ax = e, the constrained mean is:     μ_c = μ - Q⁻¹A^T(AQ⁻¹A^T)⁻¹(Aμ - e)

And samples are obtained via:     x_c = x - Q⁻¹A^T(AQ⁻¹A^T)⁻¹(Ax - e)

where x is a sample from the unconstrained distribution.

The constrained covariance matrix is:     Σ_c = Q⁻¹ - Q⁻¹A^T(AQ⁻¹A^T)⁻¹AQ⁻¹

**Implementation**

For efficiency, the constructor precomputes:
- Ã^T = Q⁻¹A^T (via solving QL^T = A^T where Q = LL^T)
  
- L_c from Cholesky factorization of AÃ^T
  
- B = L^(-T)Ã^T L_c^(-T) for variance computations
  

**Type Parameters**
- `T<:Real`: The numeric type
  
- `L<:Union{LinearMaps.LinearMap{T}, AbstractMatrix{T}}`: The precision map type
  
- `G<:AbstractGMRF{T,L}`: The concrete type of the base GMRF
  

**Fields**
- `base_gmrf::G`: The unconstrained GMRF
  
- `constraint_matrix::Matrix{T}`: Constraint matrix A (converted to dense)
  
- `constraint_vector::Vector{T}`: Constraint vector e
  
- `A_tilde_T::Matrix{T}`: Precomputed Q⁻¹A^T
  
- `L_c::Cholesky{T, Matrix{T}}`: Cholesky factorization of AÃ^T
  
- `constrained_mean::Vector{T}`: Precomputed constrained mean
  

**Constructor**

```
ConstrainedGMRF(base_gmrf::AbstractGMRF, A, e)
```


Create a constrained GMRF where `base_gmrf` is the unconstrained distribution, `A` is the constraint matrix, and `e` is the constraint vector such that Ax = e.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/arithmetic/constrained.jl#L9-L58" target="_blank" rel="noreferrer">source</a></Badge>

</details>

