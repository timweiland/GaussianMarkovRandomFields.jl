using SparseArrays
using Random
import DifferentiationInterface as DI

export ObservationLikelihood, loglik, loggrad, loghessian

"""
    ObservationLikelihood

Abstract base type for materialized observation likelihoods.

Observation likelihoods are created by materializing an observation model with specific
hyperparameters θ and observed data y. They provide efficient evaluation methods that 
only depend on the latent field x, eliminating the need to repeatedly pass θ and y.

This design provides major performance benefits in optimization loops and cleaner 
automatic differentiation boundaries.

# Usage Pattern
```julia
# Step 1: Configure observation model (factory)
obs_model = ExponentialFamily(Normal)

# Step 2: Materialize with data and hyperparameters  
obs_lik = obs_model(y; σ=1.2)

# Step 3: Fast evaluation in hot loops
ll = loglik(x, obs_lik)      # Only x argument needed!
grad = loggrad(x, obs_lik)   # Fast x-only evaluation
```

"""
abstract type ObservationLikelihood end

loglik(x, obs_lik::ObservationLikelihood) = error("loglik is not implemented for $(typeof(obs_lik)).")
(obs_lik::ObservationLikelihood)(x) = loglik(x, obs_lik)

autodiff_gradient_backend(::ObservationLikelihood) = nothing
default_hessian_backend(grad_backend) = nothing
autodiff_hessian_backend(obs_lik::ObservationLikelihood) = default_hessian_backend(autodiff_gradient_backend(obs_lik))
autodiff_gradient_prep(::ObservationLikelihood) = nothing
autodiff_hessian_prep(::ObservationLikelihood) = nothing


"""
    loggrad(x, obs_lik::ObservationLikelihood) -> Vector{Float64}

Automatic differentiation fallback for ObservationLikelihood gradient computation.
"""
function loggrad(x, obs_lik::ObservationLikelihood)
    backend = autodiff_gradient_backend(obs_lik)
    return if backend !== nothing
        grad_prep = autodiff_gradient_prep(obs_lik)
        if grad_prep !== nothing
            return DI.gradient(obs_lik, grad_prep, backend, x)
        else
            return DI.gradient(obs_lik, backend, x)
        end
    else
        obs_lik_type = typeof(obs_lik)
        error(
            "loggrad not implemented for $(obs_lik_type).\n"
                * "Try implementing `autodiff_gradient_backend(::$(obs_lik_type))`."
        )
    end
end

"""
    loghessian(x, obs_lik::ObservationLikelihood) -> AbstractMatrix{Float64}

Automatic differentiation fallback for ObservationLikelihood Hessian computation.
"""
function loghessian(x, obs_lik::ObservationLikelihood)
    backend = autodiff_hessian_backend(obs_lik)
    if backend === nothing
        backend = autodiff_gradient_backend(obs_lik)
    end
    return if backend !== nothing
        hessian_prep = autodiff_hessian_prep(obs_lik)
        if hessian_prep !== nothing
            return DI.hessian(obs_lik, hessian_prep, backend, x)
        else
            return DI.hessian(obs_lik, backend, x)
        end
    else
        obs_lik_type = typeof(obs_lik)
        error(
            "loghessian not implemented for $(obs_lik_type).\n"
                * "Try implementing `autodiff_hessian_backend(::$(obs_lik_type))`."
        )
    end
end
