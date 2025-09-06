using LinearAlgebra
using SparseArrays
using Distributions: Normal, product_distribution
import DifferentiationInterface as DI
import GaussianMarkovRandomFields: default_sparse_jacobian_backend

export NonlinearLeastSquaresModel, NonlinearLeastSquaresLikelihood

"""
    NonlinearLeastSquaresModel(f, n)

Observation model for nonlinear least squares with Gaussian noise:
    y | x ~ Normal(f(x), σ)

This model uses a Gauss–Newton approximation for the Hessian:
    ∇ℓ(x) = J(x)' (w ⊙ r),    where r = y - f(x),  w = 1 ./ σ.^2
    ∇²ℓ(x) ≈ -J(x)' Diagonal(w) J(x)

Notes
- Requires the sparse-AD extension (SparseConnectivityTracer + SparseMatrixColorings) to be loaded. If missing,
  construction or evaluation will error with a clear message.
- `f` must be out-of-place with signature `f(x)::AbstractVector` of the same length as `y`.
- `σ` can be a scalar or a vector matching `length(y)` (heteroskedastic case). It must be positive.
"""
struct NonlinearLeastSquaresModel{F} <: ObservationModel
    f::F
    n::Int
end

"""
    NonlinearLeastSquaresLikelihood <: ObservationLikelihood

Materialized likelihood for NonlinearLeastSquaresModel with precomputed weights and
cached sparse-Jacobian state. Hessian uses Gauss–Newton.
"""
struct NonlinearLeastSquaresLikelihood{F, T, JB} <: ObservationLikelihood
    f::F
    y::Vector{T}
    inv_σ²::Vector{T}
    log_const::T
    jac_backend::JB       # DI backend for sparse Jacobian
end

# -------------------------------------------------------------------------------------------------
# Factory pattern: make the model callable to materialize a likelihood
# -------------------------------------------------------------------------------------------------

function (model::NonlinearLeastSquaresModel)(y::AbstractVector; σ)
    # Validate σ and normalize to vector of inverse variances
    m = length(y)
    σv = _normalize_sigma(σ, m)
    any(σv .<= 0) && error("All σ entries must be positive.")
    inv_σ² = 1.0 ./ (σv .^ 2)

    # Precompute constant term: -m/2 * log(2π) - sum(log σ)
    log_const = -0.5 * m * log(2π) - sum(log, σv)

    # Prepare y
    y_vec = collect(Float64, y)
    T = promote_type(eltype(y_vec), eltype(inv_σ²))

    # Prepare sparse Jacobian backend via extension
    jac_backend = try
        default_sparse_jacobian_backend()
    catch err
        if err isa MethodError
            error(
                "Sparse Jacobian backend not available.\n" *
                    "Install/enable SparseConnectivityTracer and SparseMatrixColorings to activate the AutoSparse backend."
            )
        else
            rethrow()
        end
    end
    return NonlinearLeastSquaresLikelihood{typeof(model.f), T, typeof(jac_backend)}(
        model.f, y_vec, convert.(T, inv_σ²), convert(T, log_const), jac_backend,
    )
end

# -------------------------------------------------------------------------------------------------
# Observation model interface hooks
hyperparameters(::NonlinearLeastSquaresModel) = (:σ,)
latent_dimension(model::NonlinearLeastSquaresModel, y::AbstractVector) = model.n

# -------------------------------------------------------------------------------------------------
# Core likelihood API
# -------------------------------------------------------------------------------------------------

function loglik(x::AbstractVector, lik::NonlinearLeastSquaresLikelihood)
    yhat = lik.f(x)
    r = lik.y .- yhat
    sse = dot(lik.inv_σ², r .^ 2)
    return lik.log_const - 0.5 * sse
end

function loggrad(x::AbstractVector, lik::NonlinearLeastSquaresLikelihood)
    yhat = lik.f(x)
    r = lik.y .- yhat
    J = DI.jacobian(lik.f, lik.jac_backend, x)
    return J' * (lik.inv_σ² .* r)
end

function loghessian(x::AbstractVector, lik::NonlinearLeastSquaresLikelihood)
    # Gauss–Newton Hessian: -J' W J via DI sparse Jacobian
    J = DI.jacobian(lik.f, lik.jac_backend, x)
    H = -(J' * (Diagonal(lik.inv_σ²) * J))
    return Symmetric(H)
end

function conditional_distribution(model::NonlinearLeastSquaresModel, x::AbstractVector; σ)
    ŷ = model.f(x)
    if σ isa AbstractVector
        length(σ) == length(ŷ) || error("Length of σ vector must match f(x)")
        return product_distribution(Normal.(ŷ, σ))
    elseif σ isa Number
        return product_distribution(Normal.(ŷ, σ))
    else
        error("σ must be a number or a vector")
    end
end

# -------------------------------------------------------------------------------------------------
# Internals
# -------------------------------------------------------------------------------------------------

@inline function _normalize_sigma(σ, m::Integer)
    if σ isa Number
        return fill(float(σ), m)
    elseif σ isa AbstractVector
        length(σ) == m || error("Length of σ vector must match y (expected $m, got $(length(σ)))")
        return float.(σ)
    else
        error("σ must be a number or a vector")
    end
end
