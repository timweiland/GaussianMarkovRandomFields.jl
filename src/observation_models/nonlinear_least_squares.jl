using LinearAlgebra
using SparseArrays
using Distributions: Normal, product_distribution
import DifferentiationInterface as DI
import GaussianMarkovRandomFields: known_pattern_jacobian_backend

export NonlinearLeastSquaresModel, NonlinearLeastSquaresLikelihood

"""
    NonlinearLeastSquaresModel(f, n; hyperparams=())

Observation model for nonlinear least squares with Gaussian noise:
    y | x ~ Normal(f(x), σ)

This model uses a Gauss–Newton approximation for the Hessian:
    ∇ℓ(x) = J(x)' (w ⊙ r),    where r = y - f(x),  w = 1 ./ σ.^2
    ∇²ℓ(x) ≈ -J(x)' Diagonal(w) J(x)

Notes
- Requires the sparse-AD extension (SparseConnectivityTracer + SparseMatrixColorings) to be loaded. If missing,
  construction or evaluation will error with a clear message.
- `f` must be out-of-place and return an `AbstractVector` of the same length as `y`.
- `σ` can be a scalar or a vector matching `length(y)` (heteroskedastic case). It must be positive.

# Hyperparameter-dependent residual
By default the residual `f` has signature `f(x)::AbstractVector` and the only
hyperparameter is `σ`. To let `f` depend on hyperparameters θ, write it with the
signature `f(x; θ...)` and declare the names via `hyperparams`:

```julia
f(x; α) = α .* g(x) .- h(x)
model = NonlinearLeastSquaresModel(f, n; hyperparams = (:α,))
lik   = model(y; σ = 0.3, α = 1.5)   # σ → noise, α → f
hyperparameters(model) == (:σ, :α)
```

The declared θ are stored on the materialized likelihood and splatted into `f` at
evaluation time (mirroring [`AutoDiffObservationModel`](@ref)); with no declared
hyperparameters the stored `f` is called directly, so the fixed path is unchanged.

!!! note "Differentiating the hyperparameters"
    The residual Jacobian's sparsity pattern is detected once at materialization and
    reused, so `loggrad`/`loghessian` compose with an outer ForwardDiff pass.
    Forward-mode (`ForwardDiff`) hyperparameter gradients through `gaussian_approximation`
    with a `WorkspaceGMRF` prior are supported and **exact**: the implicit-function mode
    sensitivity is solved with the true Hessian — the Gauss–Newton posterior precision
    corrected by the residual-curvature term `Σₖ (W r)ₖ ∇²fₖ` — so exactness holds even
    for residuals nonlinear in `x`. The Jacobian sparsity pattern must be θ-independent
    (see the warning above).

    Reverse-mode AD (Zygote, Mooncake, …) through `gaussian_approximation` is **not**
    supported for this likelihood — it would require differentiating the sparse forward
    Jacobian, which reverse-mode backends cannot do — and raises a clear error pointing
    here. Use forward-mode for the hyperparameters.
"""
struct NonlinearLeastSquaresModel{F, H <: Tuple{Vararg{Symbol}}} <: ObservationModel
    f::F
    n::Int
    hyperparams::H
end

NonlinearLeastSquaresModel(f, n::Int; hyperparams::Tuple{Vararg{Symbol}} = ()) =
    NonlinearLeastSquaresModel(f, n, hyperparams)

"""
    NonlinearLeastSquaresLikelihood <: ObservationLikelihood

Materialized likelihood for NonlinearLeastSquaresModel with precomputed weights and
cached sparse-Jacobian state. Hessian uses Gauss–Newton. `hyperparams` holds the θ
values (if any) splatted into the residual `f(x; θ...)`; it is empty for the
non-parameterized case.
"""
struct NonlinearLeastSquaresLikelihood{F, T, JB, HP <: NamedTuple} <: ObservationLikelihood
    f::F
    y::Vector{T}
    inv_σ²::Vector{T}
    log_const::T
    jac_backend::JB       # DI backend for sparse Jacobian
    hyperparams::HP       # θ values splatted into f(x; θ...); empty NamedTuple if none
end

# Bind a residual to its hyperparameters. Empty θ ⇒ the residual is used directly
# (zero overhead, identical to the pre-θ-dependence path); otherwise a 1-arg closure
# splats θ. Used both at materialization (for sparsity detection on a primal residual)
# and at evaluation.
@inline _bind_residual(f, ::NamedTuple{(), Tuple{}}) = f
@inline _bind_residual(f, hp::NamedTuple) = x -> f(x; hp...)

_residual_function(lik::NonlinearLeastSquaresLikelihood) = _bind_residual(lik.f, lik.hyperparams)

# -------------------------------------------------------------------------------------------------
# Factory pattern: make the model callable to materialize a likelihood
# -------------------------------------------------------------------------------------------------

function (model::NonlinearLeastSquaresModel)(y::AbstractVector; σ, kwargs...)
    # Validate σ and normalize to vector of inverse variances
    m = length(y)
    σv = _normalize_sigma(σ, m)
    any(σv .<= 0) && throw(DomainError(σ, "All σ entries must be positive."))
    inv_σ² = 1.0 ./ (σv .^ 2)

    # Precompute constant term: -m/2 * log(2π) - sum(log σ)
    log_const = -0.5 * m * log(2π) - sum(log, σv)

    # Prepare y
    y_vec = collect(Float64, y)
    T = promote_type(eltype(y_vec), eltype(inv_σ²))

    # θ for the residual (empty NamedTuple when the model declares none).
    hp = _project_hyperparameters(model.hyperparams, values(kwargs))

    # Detect the Jacobian sparsity pattern once, on the *primal* residual (so
    # detection never traces AD-tagged hyperparameters), and reuse it for every
    # evaluation. The resulting backend's AutoForwardDiff inner nests cleanly under
    # an outer ForwardDiff pass, which is what makes hyperparameter gradients work.
    f_probe = _bind_residual(model.f, _strip_ad_partials_hyperparams(hp))
    # The stub throws a clear ArgumentError if SparseConnectivityTracer /
    # SparseMatrixColorings aren't loaded.
    jac_backend = known_pattern_jacobian_backend(f_probe, zeros(model.n))
    return NonlinearLeastSquaresLikelihood{typeof(model.f), T, typeof(jac_backend), typeof(hp)}(
        model.f, y_vec, convert.(T, inv_σ²), convert(T, log_const), jac_backend, hp,
    )
end

# -------------------------------------------------------------------------------------------------
# Observation model interface hooks
hyperparameters(model::NonlinearLeastSquaresModel) = _merge_hyperparameter_names((:σ,), model.hyperparams)
latent_dimension(model::NonlinearLeastSquaresModel, y::AbstractVector) = model.n

# Whether a likelihood's score/Hessian go through a Gauss–Newton sparse Jacobian.
# Reverse-mode AD through `gaussian_approximation` differentiates `loggrad`/`loghessian`,
# which for these likelihoods means differentiating a sparse forward-mode Jacobian — not
# supported by reverse-mode backends. The `gaussian_approximation` rrules check this and
# raise `_reverse_mode_gauss_newton_error` instead of failing deep in AD internals.
_has_gauss_newton_jacobian(::ObservationLikelihood) = false
_has_gauss_newton_jacobian(::NonlinearLeastSquaresLikelihood) = true
_has_gauss_newton_jacobian(lik::CompositeLikelihood) = any(_has_gauss_newton_jacobian, lik.components)
_has_gauss_newton_jacobian(lik::LinearlyTransformedLikelihood) = _has_gauss_newton_jacobian(lik.base_likelihood)

# COV_EXCL_START
function _reverse_mode_gauss_newton_error()
    throw(
        ArgumentError(
            "Reverse-mode automatic differentiation through `gaussian_approximation` is not " *
                "supported for NonlinearLeastSquares likelihoods. The Gauss–Newton score needs the " *
                "residual Jacobian, computed by a sparse forward-mode AD pass that reverse-mode " *
                "backends cannot differentiate through.\n" *
                "Use forward-mode AD (ForwardDiff) for the hyperparameters instead — it is exact and " *
                "efficient for the typically low-dimensional residual hyperparameters. Conditioning " *
                "through a `WorkspaceGMRF` prior reuses the symbolic factorization across hyperparameter values."
        )
    )
end
# COV_EXCL_STOP

# -------------------------------------------------------------------------------------------------
# Core likelihood API
# -------------------------------------------------------------------------------------------------

function loglik(x::AbstractVector, lik::NonlinearLeastSquaresLikelihood)
    f = _residual_function(lik)
    yhat = f(x)
    r = lik.y .- yhat
    sse = dot(lik.inv_σ², r .^ 2)
    return lik.log_const - 0.5 * sse
end

function loggrad(x::AbstractVector, lik::NonlinearLeastSquaresLikelihood)
    f = _residual_function(lik)
    yhat = f(x)
    r = lik.y .- yhat
    J = DI.jacobian(f, lik.jac_backend, x)
    return J' * (lik.inv_σ² .* r)
end

function loghessian(x::AbstractVector, lik::NonlinearLeastSquaresLikelihood)
    # Gauss–Newton Hessian: -J' W J via DI sparse Jacobian
    f = _residual_function(lik)
    J = DI.jacobian(f, lik.jac_backend, x)
    H = -(J' * (Diagonal(lik.inv_σ²) * J))
    return Symmetric(H)
end

function conditional_distribution(model::NonlinearLeastSquaresModel, x::AbstractVector; σ, kwargs...)
    ŷ = model.f(x; _project_hyperparameters(model.hyperparams, values(kwargs))...)
    if σ isa AbstractVector
        length(σ) == length(ŷ) || throw(DimensionMismatch("Length of σ vector ($(length(σ))) must match f(x) (got $(length(ŷ)))"))
        return product_distribution(Normal.(ŷ, σ))
    elseif σ isa Number
        return product_distribution(Normal.(ŷ, σ))
    else
        throw(ArgumentError("σ must be a number or a vector"))
    end
end

# -------------------------------------------------------------------------------------------------
# Internals
# -------------------------------------------------------------------------------------------------

@inline function _normalize_sigma(σ, m::Integer)
    if σ isa Number
        return fill(float(σ), m)
    elseif σ isa AbstractVector
        length(σ) == m || throw(DimensionMismatch("Length of σ vector must match y (expected $m, got $(length(σ)))"))
        return float.(σ)
    else
        throw(ArgumentError("σ must be a number or a vector"))
    end
end

# -------------------------------------------------------------------------------------------------
# Pointwise log-likelihood implementation
# -------------------------------------------------------------------------------------------------

"""
    _pointwise_loglik(::ConditionallyIndependent, x, lik::NonlinearLeastSquaresLikelihood) -> Vector{Float64}

Compute per-observation log-likelihoods for nonlinear least squares model.

For Gaussian observations y | x ~ Normal(f(x), σ), the pointwise log-likelihood is:
    log p(yᵢ | xᵢ) = -0.5 * log(2π) - log(σᵢ) - 0.5 * inv_σ²ᵢ * (yᵢ - f(x)ᵢ)²
"""
function _pointwise_loglik(::ConditionallyIndependent, x, lik::NonlinearLeastSquaresLikelihood)
    ŷ = _residual_function(lik)(x)
    residuals = lik.y .- ŷ

    # Compute element-wise log-likelihoods
    # log p(yᵢ | xᵢ) = -0.5*log(2π) - log(σᵢ) - 0.5*inv_σ²ᵢ*(yᵢ - ŷᵢ)²
    # We precomputed log_const = -m/2*log(2π) - sum(log σ), so per-obs constant is:
    # -0.5*log(2π) - log(σᵢ) = (log_const + 0.5*m*log(2π)) / m + individual log(σᵢ) term

    # Simpler: compute directly from inv_σ² (which is 1/σᵢ²)
    σ = 1.0 ./ sqrt.(lik.inv_σ²)
    return logpdf.(Normal.(ŷ, σ), lik.y)
end

"""
    _pointwise_loglik!(::ConditionallyIndependent, result, x, lik::NonlinearLeastSquaresLikelihood) -> Vector{Float64}

In-place pointwise log-likelihood for nonlinear least squares model.
"""
function _pointwise_loglik!(::ConditionallyIndependent, result, x, lik::NonlinearLeastSquaresLikelihood)
    ŷ = _residual_function(lik)(x)
    σ = 1.0 ./ sqrt.(lik.inv_σ²)

    @inbounds for i in eachindex(result, ŷ, σ, lik.y)
        result[i] = logpdf(Normal(ŷ[i], σ[i]), lik.y[i])
    end

    return result
end
