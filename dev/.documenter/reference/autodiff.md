
# Automatic Differentiation {#Automatic-Differentiation}

GaussianMarkovRandomFields.jl provides automatic differentiation (AD) support for parameter estimation, Bayesian inference, and optimization workflows involving GMRFs. Two complementary approaches are available, each with distinct strengths and use cases.

## Zygote with custom chainrules {#Zygote-with-custom-chainrules}

The primary AD implementation uses ChainRulesCore.jl to provide efficient reverse-mode automatic differentiation rules. Zygote.jl will automatically load and use these rules. The implementation leverages [SelectedInversion.jl](https://github.com/timweiland/SelectedInversion.jl) to compute gradients efficiently without materializing full covariance matrices. 

### Supported Operations {#Supported-Operations}

**GMRF Construction**: Differentiation through `GMRF(μ, Q, solver_blueprint)` and `GMRF(μ, Q)` constructors, enabling gradients to flow back to mean vectors and precision matrices.

**Log-probability Density**: Efficient differentiation of `logpdf(gmrf, z)` computations using selected inverses.

### Current Limitations {#Current-Limitations}

**Conditional GMRFs**: Chain rules do not currently support `condition_on_observations`. For workflows requiring conditional inference with AD, use the LDLFactorizations approach below. Contributions to extend chain rules support to conditional operations are welcome.

### Solver Compatibility {#Solver-Compatibility}

Chain rules work with any Cholesky-based solver backend:
- **CHOLMOD** (default): Fast sparse Cholesky via SuiteSparse
  
- **Pardiso**: Pardiso solver (via package extension)
  
- **LDLFactorizations**: Pure Julia implementation of a Cholesky solver
  

### Basic Usage Example {#Basic-Usage-Example}

```julia
using GaussianMarkovRandomFields, Zygote, SparseArrays

# Define precision matrix
Q = spdiagm(-1 => -0.5*ones(99), 0 => ones(100), 1 => -0.5*ones(99))

# Sample point for evaluation
z = randn(100)

# Define function to differentiate
gmrf_from_mean = θ -> GMRF(θ, Q)
logpdf_from_mean = θ -> logpdf(gmrf_from_mean(θ), z)

# Differentiate
θ_eval = zeros(100)
gradient(logpdf_from_mean, θ_eval)[1]
```


## LDLFactorizations Approach {#LDLFactorizations-Approach}

The alternative approach uses LDLFactorizations.jl, a plain Julia implementation of sparse Cholesky factorization that supports automatic differentiation through all Julia AD libraries. This &quot;just works&quot;, but you&#39;re limited to LDLFactorizations.jl. This may be fine for moderately sized problems, but CHOLMOD and Pardiso will generally scale much more efficiently.

### When to Use {#When-to-Use}
- **Conditional GMRFs**: (Currently) required for `condition_on_observations` with AD
  
- **Forward-mode AD**: Efficient for problems with few parameters
  

### Solver Configuration {#Solver-Configuration}

Use the `:autodiffable` solver variant:

```julia
# Configure autodiffable solver
blueprint = CholeskySolverBlueprint{:autodiffable}()
gmrf = GMRF(μ, Q, blueprint)
```


## Troubleshooting {#Troubleshooting}

**Error: &quot;Automatic differentiation through logpdf currently only supports Cholesky-based solvers&quot;**
- Solution: Ensure your GMRF uses a Cholesky-based solver, not CGSolver or other iterative methods
  

**Poor performance with chain rules**
- Check if you&#39;re accidentally using `:autodiffable` solver when chain rules would work with default solver
  
