import DifferentiationInterface as DI

export AutoDiffLikelihood, AutoDiffObservationModel

"""
    AutoDiffLikelihood{F, B, SB, GP, HP} <: ObservationLikelihood

Automatic differentiation-based observation likelihood that wraps a user-provided log-likelihood function.

This is a materialized likelihood created from an AutoDiffObservationModel. The log-likelihood
function is typically a closure that already includes hyperparameters and data.

# Type Parameters
- `F`: Type of the log-likelihood function (usually a closure)
- `B`: Type of the AD backend for gradients
- `SB`: Type of the AD backend for Hessians
- `GP`: Type of the gradient preparation object
- `HP`: Type of the Hessian preparation object

# Fields
- `loglik_func::F`: Log-likelihood function with signature `(x) -> Real`
- `grad_backend::B`: AD backend for gradient computation
- `hess_backend::SB`: AD backend for Hessian computation
- `grad_prep::GP`: Preparation object for gradient computation
- `hess_prep::HP`: Preparation object for Hessian computation

# Usage
Typically created via AutoDiffObservationModel factory:
```julia
# Define your log-likelihood function with hyperparameters
function my_loglik(x; σ=1.0, y=[1.0, 2.0])
    μ = x  # or some transformation of x
    return -0.5 * sum((y .- μ).^2) / σ^2 - length(y) * log(σ)
end

# Create observation model
obs_model = AutoDiffObservationModel(my_loglik, n_latent=2)

# Materialize with data and hyperparameters
obs_lik = obs_model(σ=0.5, y=[1.2, 1.8])  # Creates AutoDiffLikelihood

# Use in the standard way
x = [1.1, 1.9]
ll = loglik(x, obs_lik)
grad = loggrad(x, obs_lik)  # Uses prepared AD with optimal backends
hess = loghessian(x, obs_lik)  # Automatically sparse when available!
```

# Sparse Hessian Features
The Hessian computation automatically:
- Detects sparsity pattern using TracerSparsityDetector
- Uses greedy coloring for efficient computation
- Returns a sparse matrix when beneficial
- Falls back to dense computation for small problems

See also: [`loglik`](@ref), [`loggrad`](@ref), [`loghessian`](@ref)
"""
struct AutoDiffLikelihood{F, B, SB, GP, HP} <: GaussianMarkovRandomFields.ObservationLikelihood
    loglik_func::F
    grad_backend::B
    hess_backend::SB
    grad_prep::GP
    hess_prep::HP
end

"""
    AutoDiffObservationModel{F, B, SB, H} <: ObservationModel

Observation model that uses automatic differentiation for a user-provided log-likelihood function.

This serves as a factory for creating AutoDiffLikelihood instances. The user provides a 
log-likelihood function that can accept hyperparameters, and when materialized, creates
a closure with the hyperparameters baked in.

# Type Parameters
- `F`: Type of the log-likelihood function  
- `B`: Type of the AD backend for gradients
- `SB`: Type of the AD backend for Hessians
- `H`: Type of the hyperparameters tuple

# Fields
- `loglik_func::F`: User-provided log-likelihood function with signature `(x; kwargs...) -> Real`
- `n_latent::Int`: Number of latent field components
- `grad_backend::B`: AD backend for gradient computation
- `hess_backend::SB`: AD backend for Hessian computation
- `hyperparams::H`: Tuple of hyperparameter names that this model expects

# Usage
```julia
# Define your log-likelihood function with hyperparameters
function my_loglik(x; σ=1.0, y=[1.0, 2.0])
    μ = x  # or some transformation of x
    return -0.5 * sum((y .- μ).^2) / σ^2 - length(y) * log(σ)
end

# Create observation model specifying expected hyperparameters
obs_model = AutoDiffObservationModel(my_loglik; n_latent=2, hyperparams=(:σ, :y))

# Materialize with specific hyperparameters
obs_lik = obs_model(σ=0.5, y=[1.2, 1.8])  # Creates AutoDiffLikelihood

# Use normally
ll = loglik(x, obs_lik)
grad = loggrad(x, obs_lik)
hess = loghessian(x, obs_lik)
```

See also: [`AutoDiffLikelihood`](@ref), [`ObservationModel`](@ref)
"""
struct AutoDiffObservationModel{F, B, SB, H} <: GaussianMarkovRandomFields.ObservationModel
    loglik_func::F
    n_latent::Int
    grad_backend::B
    hess_backend::SB
    hyperparams::H
end

const AD_PREFERRED_ORDER = (DI.AutoEnzyme(), DI.AutoMooncake(), DI.AutoZygote(), DI.AutoForwardDiff())
function default_grad_backend()
    ad_idx = findfirst(DI.check_available, AD_PREFERRED_ORDER)
    if ad_idx === nothing
        error(
            "None of the default AD backends are available."
                * " Please specify your own grad_backend."
        )
    end
    return AD_PREFERRED_ORDER[ad_idx]
end

"""
    AutoDiffObservationModel(loglik_func; n_latent, hyperparams=(), grad_backend, hessian_backend) -> AutoDiffObservationModel

Construct an AutoDiffObservationModel with the given log-likelihood function and AD backends.

# Arguments
- `loglik_func`: Function with signature `(x; kwargs...) -> Real` that computes log-likelihood
- `n_latent`: Number of latent field components (required)
- `hyperparams`: Tuple of hyperparameter names that this model expects (defaults to empty tuple)
- `grad_backend`: AD backend for gradient computation (defaults to auto-detected)
- `hessian_backend`: AD backend for Hessian computation (defaults to sparse when available)

# Returns
- `AutoDiffObservationModel`: Factory for creating materialized likelihoods

# Example
```julia
function poisson_loglik(x; y=[1, 3, 0, 2])
    return sum(y .* x .- exp.(x))
end

# Model with hyperparameters
obs_model = AutoDiffObservationModel(poisson_loglik; n_latent=4, hyperparams=(:y,))
obs_lik = obs_model(y=[2, 1, 3, 0])  # Materialize with specific data

# Model without hyperparameters  
simple_loglik(x) = sum(x.^2)
simple_model = AutoDiffObservationModel(simple_loglik; n_latent=3)  # hyperparams=() by default
simple_lik = simple_model()  # No hyperparameters needed
```
"""
function AutoDiffObservationModel(loglik_func; n_latent, hyperparams = (), grad_backend = default_grad_backend(), hessian_backend = default_hessian_backend(grad_backend))
    if hessian_backend === nothing
        hessian_backend = grad_backend
        @warn "Hessian backend has type $(typeof(hessian_backend)) which may produce dense Hessians!!"
    end
    return AutoDiffObservationModel(loglik_func, n_latent, grad_backend, hessian_backend, hyperparams)
end

"""
    AutoDiffLikelihood(loglik_func; n_latent, grad_backend, hessian_backend) -> AutoDiffLikelihood

Construct an AutoDiffLikelihood directly with a log-likelihood function.

# Arguments
- `loglik_func`: Function with signature `(x) -> Real` that computes log-likelihood
- `n_latent`: Number of latent field components (required for preparation)
- `grad_backend`: AD backend for gradient computation (defaults to auto-detected)
- `hessian_backend`: AD backend for Hessian computation (defaults to sparse when available)

# Returns
- `AutoDiffLikelihood`: Ready-to-use likelihood with prepared AD backends

# Example
```julia
# Directly construct (more common to use AutoDiffObservationModel)
poisson_loglik(x) = sum([1, 3, 0, 2] .* x .- exp.(x))  # Closure with data
obs_lik = AutoDiffLikelihood(poisson_loglik; n_latent=4)
```
"""
function AutoDiffLikelihood(loglik_func; n_latent, grad_backend = default_grad_backend(), hessian_backend = default_hessian_backend(grad_backend))
    if hessian_backend === nothing
        hessian_backend = grad_backend
        @warn "Hessian backend has type $(typeof(hessian_backend)) which may produce dense Hessians!!"
    end
    x_proto = zeros(n_latent)
    grad_prep = DI.prepare_gradient(loglik_func, grad_backend, x_proto)
    hess_prep = DI.prepare_hessian(loglik_func, hessian_backend, x_proto)
    return AutoDiffLikelihood(loglik_func, grad_backend, hessian_backend, grad_prep, hess_prep)
end

function (obs_model::AutoDiffObservationModel)(y; kwargs...)
    # Create a closure with hyperparameters baked in
    closure = x -> obs_model.loglik_func(x; y, kwargs...)

    # Create AutoDiffLikelihood with the closure
    return AutoDiffLikelihood(
        closure;
        n_latent = obs_model.n_latent,
        grad_backend = obs_model.grad_backend,
        hessian_backend = obs_model.hess_backend
    )
end

# =======================================================================================
# OBSERVATION MODEL INTERFACE IMPLEMENTATION
# =======================================================================================
latent_dimension(obs_model::AutoDiffObservationModel, y::AbstractVector) = obs_model.n_latent
hyperparameters(obs_model::AutoDiffObservationModel) = obs_model.hyperparams

# =======================================================================================
# CORE LIKELIHOOD EVALUATION METHOD
# =======================================================================================

"""
    loglik(x, obs_lik::AutoDiffLikelihood) -> Real

Evaluate the log-likelihood function at latent field `x`.

Calls the stored log-likelihood function, which typically includes all necessary
hyperparameters and data as a closure.
"""
function loglik(x, obs_lik::AutoDiffLikelihood)
    return obs_lik.loglik_func(x)
end

function loggrad(x, obs_lik::AutoDiffLikelihood)
    return DI.gradient(obs_lik.loglik_func, obs_lik.grad_prep, obs_lik.grad_backend, x)
end

function loghessian(x, obs_lik::AutoDiffLikelihood)
    return DI.hessian(obs_lik.loglik_func, obs_lik.hess_prep, obs_lik.hess_backend, x)
end

# =======================================================================================
# AUTODIFF INTERFACE IMPLEMENTATION
# =======================================================================================
autodiff_gradient_backend(obs_lik::AutoDiffLikelihood) = obs_lik.grad_backend
autodiff_hessian_backend(obs_lik::AutoDiffLikelihood) = obs_lik.hess_backend
autodiff_gradient_prep(obs_lik::AutoDiffLikelihood) = obs_lik.grad_prep
autodiff_hessian_prep(obs_lik::AutoDiffLikelihood) = obs_lik.hess_prep

# =======================================================================================
# PRETTY PRINTING
# =======================================================================================

# COV_EXCL_START
function Base.show(io::IO, obs_model::AutoDiffObservationModel)
    func_name = string(obs_model.loglik_func)
    # Try to extract a cleaner function name if possible
    if occursin("var\"", func_name) || occursin("#", func_name)
        func_name = "user function"
    end

    hyperparams_str = isempty(obs_model.hyperparams) ? "none" : join(obs_model.hyperparams, ", ")

    print(io, "AutoDiffObservationModel(")
    print(io, func_name)
    print(io, "; n_latent=", obs_model.n_latent)
    print(io, ", hyperparams=(", hyperparams_str, ")")
    print(io, ", grad_backend=", Base.typename(typeof(obs_model.grad_backend)).wrapper)
    print(io, ", hess_backend=", Base.typename(typeof(obs_model.hess_backend)).wrapper)
    return print(io, ")")
end

function Base.show(io::IO, obs_lik::AutoDiffLikelihood)
    func_name = string(obs_lik.loglik_func)
    # Try to extract a cleaner function name if possible
    if occursin("var\"", func_name) || occursin("#", func_name)
        func_name = "closure"
    end

    print(io, "AutoDiffLikelihood(")
    print(io, func_name)
    print(io, "; grad_backend=", Base.typename(typeof(obs_lik.grad_backend)).wrapper)
    print(io, ", hess_backend=", Base.typename(typeof(obs_lik.hess_backend)).wrapper)
    return print(io, ")")
end
# COV_EXCL_STOP
