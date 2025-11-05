using LinearAlgebra
using SparseArrays

export LinearlyTransformedObservationModel, LinearlyTransformedLikelihood

"""
    LinearlyTransformedObservationModel{M, A} <: ObservationModel

Observation model that applies a linear transformation to the latent field before 
passing to a base observation model. This enables GLM-style modeling with design 
matrices while maintaining full compatibility with existing observation models.

# Mathematical Foundation
The wrapper transforms the full latent field x_full to linear predictors η via a 
design matrix A:
- η = A * x_full  
- Base model operates on η as usual: p(y | η, θ)
- Chain rule applied for gradients/Hessians: 
  - ∇_{x_full} ℓ = A^T ∇_η ℓ
  - ∇²_{x_full} ℓ = A^T ∇²_η ℓ A

# Type Parameters
- `M <: ObservationModel`: Type of the base observation model
- `A`: Type of the design matrix (typically AbstractMatrix)

# Fields
- `base_model::M`: The underlying observation model that operates on linear predictors
- `design_matrix::A`: Matrix mapping full latent field to observation-specific linear predictors

# Usage Pattern
```julia
# Step 1: Create base observation model
base_model = ExponentialFamily(Poisson)  # LogLink by default

# Step 2: Create design matrix (maps latent field to linear predictors)
# For: y ~ intercept + temperature + group_effects
A = [1.0  20.0  1.0  0.0  0.0;   # obs 1: intercept + temp + group1
     1.0  25.0  1.0  0.0  0.0;   # obs 2: intercept + temp + group1  
     1.0  30.0  0.0  1.0  0.0;   # obs 3: intercept + temp + group2
     1.0  15.0  0.0  0.0  1.0]   # obs 4: intercept + temp + group3

# Step 3: Create wrapped model
obs_model = LinearlyTransformedObservationModel(base_model, A)

# Step 4: Use in GMRF model - latent field now includes all components
# x_full = [β₀, β₁, u₁, u₂, u₃]  # intercept, slope, group effects

# Step 5: Materialize with data and hyperparameters
obs_lik = obs_model(y; σ=1.2)  # Creates LinearlyTransformedLikelihood

# Step 6: Fast evaluation in optimization loops
ll = loglik(x_full, obs_lik)
```

# Hyperparameters
All hyperparameters come from the base observation model. The design matrix 
introduces no new hyperparameters - it's a fixed linear transformation.

See also: [`LinearlyTransformedLikelihood`](@ref), [`ExponentialFamily`](@ref), [`ObservationModel`](@ref)
"""
struct LinearlyTransformedObservationModel{M <: ObservationModel, A} <: ObservationModel
    base_model::M
    design_matrix::A

    function LinearlyTransformedObservationModel(base_model::M, design_matrix::A) where {M <: ObservationModel, A}
        # Validate that design matrix is appropriate
        if size(design_matrix, 1) == 0
            error("Design matrix must have at least one row (observation)")
        end
        if size(design_matrix, 2) == 0
            error("Design matrix must have at least one column (latent component)")
        end
        if design_matrix isa Matrix
            @warn "Received a dense design matrix. This can lead to major performance bottlenecks!"
        end

        return new{M, A}(base_model, design_matrix)
    end
end

"""
    LinearlyTransformedLikelihood{L, A} <: ObservationLikelihood

Materialized likelihood for LinearlyTransformedObservationModel with precomputed 
base likelihood and design matrix.

This is created by calling a LinearlyTransformedObservationModel instance with 
data and hyperparameters, following the factory pattern used throughout the package.

# Type Parameters
- `L <: ObservationLikelihood`: Type of the materialized base likelihood
- `A`: Type of the design matrix

# Fields
- `base_likelihood::L`: Materialized base observation likelihood (contains y and θ)
- `design_matrix::A`: Design matrix mapping full latent field to linear predictors

# Usage
This type is typically created automatically:
```julia
ltom = LinearlyTransformedObservationModel(base_model, design_matrix)
ltlik = ltom(y; σ=1.2)  # Creates LinearlyTransformedLikelihood
ll = loglik(x_full, ltlik)  # Fast evaluation
```
"""
struct LinearlyTransformedLikelihood{L <: ObservationLikelihood, A} <: ObservationLikelihood
    base_likelihood::L
    design_matrix::A
end

# =======================================================================================
# FACTORY PATTERN: Make LinearlyTransformedObservationModel callable
# =======================================================================================

function (ltom::LinearlyTransformedObservationModel)(y; kwargs...)
    # Create materialized base likelihood
    base_likelihood = ltom.base_model(y; kwargs...)

    # Wrap with design matrix
    return LinearlyTransformedLikelihood(base_likelihood, ltom.design_matrix)
end

# =======================================================================================
# HYPERPARAMETER INTERFACE DELEGATION
# =======================================================================================

hyperparameters(ltom::LinearlyTransformedObservationModel) = hyperparameters(ltom.base_model)
latent_dimension(ltom::LinearlyTransformedObservationModel, y::AbstractVector) = size(ltom.design_matrix, 2)

# =======================================================================================
# CORE LIKELIHOOD EVALUATION METHODS
# =======================================================================================

function loglik(x_full, ltlik::LinearlyTransformedLikelihood)
    η = ltlik.design_matrix * x_full
    return loglik(η, ltlik.base_likelihood)
end

function loggrad(x_full, ltlik::LinearlyTransformedLikelihood)
    η = ltlik.design_matrix * x_full
    grad_η = loggrad(η, ltlik.base_likelihood)
    return ltlik.design_matrix' * grad_η  # Chain rule: A^T * grad_η
end

function loghessian(x_full, ltlik::LinearlyTransformedLikelihood)
    η = ltlik.design_matrix * x_full
    hess_η = loghessian(η, ltlik.base_likelihood)
    A = ltlik.design_matrix

    # Chain rule: A^T * hess_η * A
    # This preserves sparsity patterns efficiently
    return Symmetric(A' * hess_η * A)
end

"""
    _pointwise_loglik(::ConditionallyIndependent, x_full, ltlik::LinearlyTransformedLikelihood) -> Vector{Float64}

Compute pointwise log-likelihood by transforming to linear predictors and delegating to base likelihood.

The design matrix A maps the full latent field to observation-specific linear predictors:
η = A * x_full, then pointwise log-likelihoods are computed from the base likelihood.
"""
function _pointwise_loglik(::ConditionallyIndependent, x_full, ltlik::LinearlyTransformedLikelihood)
    η = ltlik.design_matrix * x_full
    return pointwise_loglik(η, ltlik.base_likelihood)
end

"""
    _pointwise_loglik!(::ConditionallyIndependent, result, x_full, ltlik::LinearlyTransformedLikelihood) -> Vector{Float64}

In-place pointwise log-likelihood for linearly transformed observations.

Computes η = A * x_full and delegates to the base likelihood's in-place method.
Note: Allocates temporary storage for η (cannot avoid this allocation).
"""
function _pointwise_loglik!(::ConditionallyIndependent, result, x_full, ltlik::LinearlyTransformedLikelihood)
    η = ltlik.design_matrix * x_full  # Allocates - unavoidable
    return pointwise_loglik!(result, η, ltlik.base_likelihood)
end

# =======================================================================================
# CONDITIONAL DISTRIBUTION INTERFACE
# =======================================================================================

function conditional_distribution(ltom::LinearlyTransformedObservationModel, x_full; kwargs...)
    η = ltom.design_matrix * x_full
    return conditional_distribution(ltom.base_model, η; kwargs...)
end

# COV_EXCL_START
function Base.show(io::IO, model::LinearlyTransformedObservationModel)
    A = model.design_matrix
    m, n = size(A)
    A_kind = A isa SparseArrays.AbstractSparseMatrix ? "sparse" : "dense"
    return print(io, "LinearlyTransformedObservationModel(base=$(typeof(model.base_model)), A=$(m)×$(n) $(A_kind))")
end
# COV_EXCL_STOP
