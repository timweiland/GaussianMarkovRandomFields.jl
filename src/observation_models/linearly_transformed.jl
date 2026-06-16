using LinearAlgebra
using SparseArrays

export LinearlyTransformedObservationModel, LinearlyTransformedLikelihood, ParameterizedMatrix, ParameterizedOffset

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
    ParameterizedOffset(builder; hyperparameters::Tuple{Vararg{Symbol}})

A hyperparameter-dependent additive offset `b` for [`LinearlyTransformedObservationModel`](@ref),
making the linear predictor affine: `η = A·x + b`.

`builder` is a keyword-callable returning the concrete offset vector when invoked with the
hyperparameters it declares: `builder(; θ...) -> AbstractVector`. `hyperparameters` lists the
θ-names `builder` consumes; they are merged into the model's `hyperparameters` and routed to
`builder` when the model is materialized.

The offset is resolved once, inside `model(y; θ...)`, so the materialized
[`LinearlyTransformedLikelihood`](@ref) and all downstream conditioning see only a concrete
vector — there is no per-evaluation overhead beyond the single `builder` call.

# Example
```julia
# Observing a PDE residual ℒu − S(θ) at collocation points: affine in u, with a
# hyperparameter-dependent forcing offset b = −S(θ).
build_b(; s) = -source_term(s)            # length fixed; only values depend on s
ltom = LinearlyTransformedObservationModel(
    ExponentialFamily(Normal), ℒ_matrix;
    offset = ParameterizedOffset(build_b; hyperparameters = (:s,)),
)
ltlik = ltom(y; σ = 1.0, s = 0.3)         # σ → base model, s → build_b
```

!!! warning "θ-independent length"
    The offset's *values* may depend on the hyperparameters, but its *length* may not — it
    must match the number of observations so the workspace symbolic factorization stays valid.

See also: [`LinearlyTransformedObservationModel`](@ref), [`ParameterizedMatrix`](@ref).
"""
struct ParameterizedOffset{B, H <: Tuple{Vararg{Symbol}}}
    builder::B
    hp_names::H
end

ParameterizedOffset(builder; hyperparameters::Tuple{Vararg{Symbol}}) =
    ParameterizedOffset(builder, hyperparameters)

# Resolve a (possibly parameterized) offset at materialization time. `nothing` (no offset)
# and a fixed vector resolve to themselves (identity, selected by dispatch → no runtime
# branch); a ParameterizedOffset is built once.
@inline _resolve_offset(::Nothing, ::NamedTuple) = nothing
@inline _resolve_offset(b::AbstractVector, ::NamedTuple) = b
function _resolve_offset(spec::ParameterizedOffset, θ::NamedTuple)
    return spec.builder(; _project_hyperparameters(spec.hp_names, θ)...)
end

# Hyperparameter names contributed by the offset.
_offset_hp_names(::Nothing) = ()
_offset_hp_names(::AbstractVector) = ()
_offset_hp_names(spec::ParameterizedOffset) = spec.hp_names

# Affine linear predictor η = A·x (+ b). The `Nothing` method is the pure-linear path,
# byte-for-byte the pre-offset behaviour with no allocation for an absent offset.
@inline _linear_predictor(A, x, ::Nothing) = A * x
@inline _linear_predictor(A, x, b::AbstractVector) = A * x .+ b

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
- `A`: Type of the design matrix (`AbstractMatrix` or [`ParameterizedMatrix`](@ref))
- `B`: Type of the offset (`Nothing`, `AbstractVector`, or [`ParameterizedOffset`](@ref))

# Fields
- `base_model::M`: The underlying observation model that operates on linear predictors
- `design_matrix::A`: Matrix mapping full latent field to observation-specific linear predictors
- `offset::B`: Optional additive offset `b` making the predictor affine (`η = A·x + b`); `nothing` for no offset

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

# Affine offset
The predictor may be made affine, `η = A·x + b`, via the `offset` keyword: a fixed
vector or a [`ParameterizedOffset`](@ref) builder. The offset is resolved at
materialization; `nothing` (the default) is the pure-linear path with no overhead.

# Hyperparameters
By default all hyperparameters come from the base observation model and the design
matrix introduces none — it's a fixed linear transformation. To let the design matrix
or the offset depend on hyperparameters, pass a [`ParameterizedMatrix`](@ref) and/or a
[`ParameterizedOffset`](@ref); their declared hyperparameters are merged into
`hyperparameters(model)` and resolved once at materialization.

See also: [`LinearlyTransformedLikelihood`](@ref), [`ParameterizedMatrix`](@ref), [`ParameterizedOffset`](@ref), [`ExponentialFamily`](@ref), [`ObservationModel`](@ref)
"""
struct LinearlyTransformedObservationModel{M <: ObservationModel, A, B} <: ObservationModel
    base_model::M
    design_matrix::A
    offset::B
end

# Validate the offset spec at construction: must be nothing, a fixed vector, or a
# ParameterizedOffset. (Length is checked at materialization against the design matrix.)
_validate_offset(::Nothing) = nothing
_validate_offset(::AbstractVector) = nothing
_validate_offset(::ParameterizedOffset) = nothing
# COV_EXCL_START
_validate_offset(offset) = throw(
    ArgumentError(
        "offset must be `nothing`, an `AbstractVector`, or a `ParameterizedOffset`; got $(typeof(offset))."
    )
)
# COV_EXCL_STOP

function LinearlyTransformedObservationModel(base_model::ObservationModel, design_matrix::AbstractMatrix; offset = nothing)
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
    _validate_offset(offset)
    return LinearlyTransformedObservationModel{typeof(base_model), typeof(design_matrix), typeof(offset)}(base_model, design_matrix, offset)
end

# Parameterized design matrix: no shape validation possible until the matrix is
# built (deferred to materialization); just store the spec.
function LinearlyTransformedObservationModel(base_model::ObservationModel, design_matrix::ParameterizedMatrix; offset = nothing)
    _validate_offset(offset)
    return LinearlyTransformedObservationModel{typeof(base_model), typeof(design_matrix), typeof(offset)}(base_model, design_matrix, offset)
end

"""
    LinearlyTransformedLikelihood{L, A} <: ObservationLikelihood

Materialized likelihood for LinearlyTransformedObservationModel with precomputed 
base likelihood and design matrix.

This is created by calling a LinearlyTransformedObservationModel instance with 
data and hyperparameters, following the factory pattern used throughout the package.

# Type Parameters
- `L <: ObservationLikelihood`: Type of the materialized base likelihood
- `A`: Type of the (resolved, concrete) design matrix
- `B`: Type of the (resolved) offset — `Nothing` or a concrete `AbstractVector`

# Fields
- `base_likelihood::L`: Materialized base observation likelihood (contains y and θ)
- `design_matrix::A`: Design matrix mapping full latent field to linear predictors
- `offset::B`: Additive offset `b` (`η = A·x + b`), or `nothing` for the pure-linear `η = A·x`

# Usage
This type is typically created automatically:
```julia
ltom = LinearlyTransformedObservationModel(base_model, design_matrix)
ltlik = ltom(y; σ=1.2)  # Creates LinearlyTransformedLikelihood
ll = loglik(x_full, ltlik)  # Fast evaluation
```
"""
struct LinearlyTransformedLikelihood{L <: ObservationLikelihood, A, B} <: ObservationLikelihood
    base_likelihood::L
    design_matrix::A
    offset::B
end

# Backward-compatible constructor: no offset (η = A·x).
LinearlyTransformedLikelihood(base_likelihood::ObservationLikelihood, design_matrix) =
    LinearlyTransformedLikelihood(base_likelihood, design_matrix, nothing)

# =======================================================================================
# FACTORY PATTERN: Make LinearlyTransformedObservationModel callable
# =======================================================================================

function (ltom::LinearlyTransformedObservationModel)(y; kwargs...)
    # Create materialized base likelihood (base model picks the kwargs it needs).
    base_likelihood = ltom.base_model(y; kwargs...)

    # Resolve the design matrix and offset at θ (identity for fixed objects; one
    # builder call for a ParameterizedMatrix / ParameterizedOffset) and wrap.
    θ_nt = values(kwargs)
    A = _resolve_design_matrix(ltom.design_matrix, θ_nt)
    b = _resolve_offset(ltom.offset, θ_nt)
    _check_offset_length(b, A)
    return LinearlyTransformedLikelihood(base_likelihood, A, b)
end

# The offset length must match the number of observations (rows of A); this is the
# `θ-independent length` contract surfaced at materialization with a clear message.
_check_offset_length(::Nothing, _) = nothing
function _check_offset_length(b::AbstractVector, A::AbstractMatrix)
    length(b) == size(A, 1) || throw(
        DimensionMismatch(
            "offset has length $(length(b)) but the design matrix has $(size(A, 1)) rows (observations)."
        )
    )
    return nothing
end
_check_offset_length(::AbstractVector, _) = nothing  # non-matrix A (e.g. UniformScaling): skip

# =======================================================================================
# HYPERPARAMETER INTERFACE DELEGATION
# =======================================================================================

hyperparameters(ltom::LinearlyTransformedObservationModel) =
    _merge_hyperparameter_names(
    _merge_hyperparameter_names(hyperparameters(ltom.base_model), _design_matrix_hp_names(ltom.design_matrix)),
    _offset_hp_names(ltom.offset),
)
latent_dimension(ltom::LinearlyTransformedObservationModel, y::AbstractVector) = _design_matrix_latent_dim(ltom.design_matrix)

# =======================================================================================
# CORE LIKELIHOOD EVALUATION METHODS
# =======================================================================================

function loglik(x_full, ltlik::LinearlyTransformedLikelihood)
    η = _linear_predictor(ltlik.design_matrix, x_full, ltlik.offset)
    return loglik(η, ltlik.base_likelihood)
end

function loggrad(x_full, ltlik::LinearlyTransformedLikelihood)
    η = _linear_predictor(ltlik.design_matrix, x_full, ltlik.offset)
    grad_η = loggrad(η, ltlik.base_likelihood)
    return ltlik.design_matrix' * grad_η  # Chain rule: A^T * grad_η (offset is x-independent)
end

function loghessian(x_full, ltlik::LinearlyTransformedLikelihood)
    η = _linear_predictor(ltlik.design_matrix, x_full, ltlik.offset)
    hess_η = loghessian(η, ltlik.base_likelihood)
    A = ltlik.design_matrix

    # Chain rule: H = Aᵀ·hess_η·A. η = A·x + b is affine in x, so the additive offset
    # leaves ∂η/∂x = A and the Hessian structure unchanged (it only shifts the point η at
    # which the base Hessian is evaluated).
    #
    # Parenthesize as Aᵀ·(hess_η·A), NOT (Aᵀ·hess_η)·A. On Julia ≥1.11 the left-associated
    # `Aᵀ·hess_η` dispatches to `Adjoint{SparseMatrixCSC}·Diagonal`, which materializes a
    # *fully dense* structural pattern of explicit zeros (a diagonal A blows up to n²
    # stored entries — O(n²) memory/compute, and the spurious zeros trip the workspace
    # Newton step's `pattern(H) ⊆ pattern(Q)` check). Evaluating `hess_η·A` first keeps
    # every factor a genuine sparse product, so H carries only true nonzeros.
    return Symmetric(A' * (hess_η * A))
end

"""
    _pointwise_loglik(::ConditionallyIndependent, x_full, ltlik::LinearlyTransformedLikelihood) -> Vector{Float64}

Compute pointwise log-likelihood by transforming to linear predictors and delegating to base likelihood.

The design matrix A maps the full latent field to observation-specific linear predictors:
η = A * x_full, then pointwise log-likelihoods are computed from the base likelihood.
"""
function _pointwise_loglik(::ConditionallyIndependent, x_full, ltlik::LinearlyTransformedLikelihood)
    η = _linear_predictor(ltlik.design_matrix, x_full, ltlik.offset)
    return pointwise_loglik(η, ltlik.base_likelihood)
end

"""
    _pointwise_loglik!(::ConditionallyIndependent, result, x_full, ltlik::LinearlyTransformedLikelihood) -> Vector{Float64}

In-place pointwise log-likelihood for linearly transformed observations.

Computes η = A * x_full and delegates to the base likelihood's in-place method.
Note: Allocates temporary storage for η (cannot avoid this allocation).
"""
function _pointwise_loglik!(::ConditionallyIndependent, result, x_full, ltlik::LinearlyTransformedLikelihood)
    η = _linear_predictor(ltlik.design_matrix, x_full, ltlik.offset)  # Allocates - unavoidable
    return pointwise_loglik!(result, η, ltlik.base_likelihood)
end

# =======================================================================================
# CONDITIONAL DISTRIBUTION INTERFACE
# =======================================================================================

function conditional_distribution(ltom::LinearlyTransformedObservationModel, x_full; kwargs...)
    θ_nt = values(kwargs)
    A = _resolve_design_matrix(ltom.design_matrix, θ_nt)
    b = _resolve_offset(ltom.offset, θ_nt)
    η = _linear_predictor(A, x_full, b)
    return conditional_distribution(ltom.base_model, η; kwargs...)
end

# COV_EXCL_START
_offset_descr(::Nothing) = "none"
_offset_descr(b::AbstractVector) = "fixed($(length(b)))"
_offset_descr(spec::ParameterizedOffset) = "parameterized($(isempty(spec.hp_names) ? "none" : join(spec.hp_names, ", ")))"

function Base.show(io::IO, model::LinearlyTransformedObservationModel)
    A = model.design_matrix
    A_str = if A isa ParameterizedMatrix
        "parameterized($(isempty(A.hp_names) ? "none" : join(A.hp_names, ", ")))"
    else
        m, n = size(A)
        "$(m)×$(n) $(A isa SparseArrays.AbstractSparseMatrix ? "sparse" : "dense")"
    end
    return print(io, "LinearlyTransformedObservationModel(base=$(typeof(model.base_model)), A=$(A_str), offset=$(_offset_descr(model.offset)))")
end
# COV_EXCL_STOP
