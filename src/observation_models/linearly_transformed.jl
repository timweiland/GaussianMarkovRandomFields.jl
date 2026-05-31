using LinearAlgebra
using SparseArrays

export LinearlyTransformedObservationModel, LinearlyTransformedLikelihood, ParameterizedMatrix

"""
    ParameterizedMatrix(builder; hyperparameters::Tuple{Vararg{Symbol}}, n_latent=nothing)

A hyperparameter-dependent design matrix for [`LinearlyTransformedObservationModel`](@ref).

`builder` is a keyword-callable returning the concrete design matrix when invoked with
the hyperparameters it declares: `builder(; θ...) -> AbstractMatrix`. `hyperparameters`
lists the θ-names `builder` consumes; they are merged into the model's `hyperparameters`
and routed to `builder` when the model is materialized. `n_latent`, if given, is the
number of latent components (columns) of the built matrix and is reported by
`latent_dimension`; it defaults to `nothing` (unknown until the matrix is built).

The matrix is resolved once, inside `model(y; θ...)`, so the materialized
[`LinearlyTransformedLikelihood`](@ref) and all downstream conditioning see only a
concrete matrix — there is no per-evaluation overhead beyond the single `builder` call.

# Example
```julia
build_A(; ρ) = sparse([1.0 ρ; 0.0 1.0])   # pattern fixed; only values depend on ρ
ltom = LinearlyTransformedObservationModel(
    ExponentialFamily(Normal),
    ParameterizedMatrix(build_A; hyperparameters = (:ρ,), n_latent = 2),
)
ltlik = ltom(y; σ = 1.0, ρ = 0.3)          # σ → base model, ρ → build_A
```

!!! warning "θ-independent sparsity pattern"
    The *sparsity pattern* of the returned matrix must not depend on the hyperparameters —
    only the numeric nonzero values may. Workspace-based pipelines reuse a one-time
    symbolic factorization fixed at the reference hyperparameters; a θ-varying pattern
    violates that contract and will error when the pattern changes.

See also: [`LinearlyTransformedObservationModel`](@ref).
"""
struct ParameterizedMatrix{B, H <: Tuple{Vararg{Symbol}}, N <: Union{Int, Nothing}}
    builder::B
    hp_names::H
    n_latent::N
end

ParameterizedMatrix(builder; hyperparameters::Tuple{Vararg{Symbol}}, n_latent::Union{Int, Nothing} = nothing) =
    ParameterizedMatrix(builder, hyperparameters, n_latent)

# Resolve a (possibly parameterized) design matrix at materialization time.
# The `AbstractMatrix` method is the identity, selected by dispatch so the fixed
# path compiles to a plain field read with no runtime branch.
@inline _resolve_design_matrix(A::AbstractMatrix, ::NamedTuple) = A
function _resolve_design_matrix(spec::ParameterizedMatrix, θ::NamedTuple)
    return spec.builder(; _project_hyperparameters(spec.hp_names, θ)...)
end

# Hyperparameter names / latent dimension contributed by the design matrix.
_design_matrix_hp_names(::AbstractMatrix) = ()
_design_matrix_hp_names(spec::ParameterizedMatrix) = spec.hp_names
_design_matrix_latent_dim(A::AbstractMatrix) = size(A, 2)
_design_matrix_latent_dim(spec::ParameterizedMatrix) = spec.n_latent

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
By default all hyperparameters come from the base observation model and the design
matrix introduces none — it's a fixed linear transformation. To let the design matrix
depend on hyperparameters, pass a [`ParameterizedMatrix`](@ref) instead of a plain
matrix; its declared hyperparameters are merged into `hyperparameters(model)` and the
concrete matrix is built once at materialization.

See also: [`LinearlyTransformedLikelihood`](@ref), [`ParameterizedMatrix`](@ref), [`ExponentialFamily`](@ref), [`ObservationModel`](@ref)
"""
struct LinearlyTransformedObservationModel{M <: ObservationModel, A} <: ObservationModel
    base_model::M
    design_matrix::A
end

function LinearlyTransformedObservationModel(base_model::ObservationModel, design_matrix::AbstractMatrix)
    # Validate that design matrix is appropriate
    # COV_EXCL_START
    if size(design_matrix, 1) == 0
        throw(ArgumentError("Design matrix must have at least one row (observation)"))
    end
    if size(design_matrix, 2) == 0
        throw(ArgumentError("Design matrix must have at least one column (latent component)"))
    end
    # COV_EXCL_STOP
    if design_matrix isa Matrix
        @warn "Received a dense design matrix. This can lead to major performance bottlenecks!"
    end
    return LinearlyTransformedObservationModel{typeof(base_model), typeof(design_matrix)}(base_model, design_matrix)
end

# Parameterized design matrix: no shape validation possible until the matrix is
# built (deferred to materialization); just store the spec.
LinearlyTransformedObservationModel(base_model::ObservationModel, design_matrix::ParameterizedMatrix) =
    LinearlyTransformedObservationModel{typeof(base_model), typeof(design_matrix)}(base_model, design_matrix)

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
    # Create materialized base likelihood (base model picks the kwargs it needs).
    base_likelihood = ltom.base_model(y; kwargs...)

    # Resolve the design matrix at θ (identity for a plain matrix; one builder
    # call for a ParameterizedMatrix) and wrap.
    A = _resolve_design_matrix(ltom.design_matrix, values(kwargs))
    return LinearlyTransformedLikelihood(base_likelihood, A)
end

# =======================================================================================
# HYPERPARAMETER INTERFACE DELEGATION
# =======================================================================================

hyperparameters(ltom::LinearlyTransformedObservationModel) =
    _merge_hyperparameter_names(hyperparameters(ltom.base_model), _design_matrix_hp_names(ltom.design_matrix))
latent_dimension(ltom::LinearlyTransformedObservationModel, y::AbstractVector) = _design_matrix_latent_dim(ltom.design_matrix)

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
    A = _resolve_design_matrix(ltom.design_matrix, values(kwargs))
    η = A * x_full
    return conditional_distribution(ltom.base_model, η; kwargs...)
end

# COV_EXCL_START
function Base.show(io::IO, model::LinearlyTransformedObservationModel)
    A = model.design_matrix
    if A isa ParameterizedMatrix
        names = isempty(A.hp_names) ? "none" : join(A.hp_names, ", ")
        return print(io, "LinearlyTransformedObservationModel(base=$(typeof(model.base_model)), A=parameterized($(names)))")
    end
    m, n = size(A)
    A_kind = A isa SparseArrays.AbstractSparseMatrix ? "sparse" : "dense"
    return print(io, "LinearlyTransformedObservationModel(base=$(typeof(model.base_model)), A=$(m)×$(n) $(A_kind))")
end
# COV_EXCL_STOP
