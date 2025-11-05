"""
Evaluation methods for CompositeLikelihood.

These methods implement the core mathematical operations by summing contributions
from all component likelihoods.
"""

"""
    loglik(x, composite_lik::CompositeLikelihood) -> Float64

Compute the log-likelihood of a composite likelihood by summing component contributions.

Each component likelihood receives the full latent field `x` and contributes to the total
log-likelihood. This handles cases where components may have overlapping dependencies
on the latent field.
"""
function loglik(x, composite_lik::CompositeLikelihood)
    return sum(comp -> loglik(x, comp), composite_lik.components)
end

"""
    loggrad(x, composite_lik::CompositeLikelihood) -> Vector{Float64}

Compute the gradient of the log-likelihood by summing component gradients.

Each component contributes its gradient with respect to the full latent field `x`.
For overlapping dependencies, gradients are automatically summed at each latent field element.
"""
function loggrad(x, composite_lik::CompositeLikelihood)
    # Start with zero gradient
    grad = zeros(eltype(x), length(x))

    # Sum contributions from each component
    for component in composite_lik.components
        grad .+= loggrad(x, component)
    end

    return grad
end

"""
    loghessian(x, composite_lik::CompositeLikelihood) -> AbstractMatrix{Float64}

Compute the Hessian of the log-likelihood by summing component Hessians.

Each component contributes its Hessian with respect to the full latent field `x`.
For overlapping dependencies, Hessians are automatically summed element-wise.
"""
function loghessian(x, composite_lik::CompositeLikelihood)
    # Start with zero Hessian - let first component determine type/structure
    first_hess = loghessian(x, composite_lik.components[1])
    total_hess = copy(first_hess)

    # Sum contributions from remaining components
    for i in 2:length(composite_lik.components)
        total_hess .+= loghessian(x, composite_lik.components[i])
    end

    return total_hess
end

"""
    _pointwise_loglik(::ConditionallyIndependent, x, composite_lik::CompositeLikelihood) -> Vector{Float64}

Compute pointwise log-likelihood by concatenating contributions from all components.

Each component must have conditionally independent observations. The result is a vector
containing all per-observation log-likelihoods from all components concatenated in order.

# Errors
Throws an error if any component has `ConditionallyDependent` trait.
"""
function _pointwise_loglik(::ConditionallyIndependent, x, composite_lik::CompositeLikelihood)
    # Check all components are conditionally independent
    for comp in composite_lik.components
        if observation_independence(comp) != ConditionallyIndependent()
            error(
                "CompositeLikelihood contains component with ConditionallyDependent trait.\n"
                    * "All components must have ConditionallyIndependent observations for pointwise_loglik."
            )
        end
    end

    # Concatenate pointwise log-likelihoods from all components
    return vcat([pointwise_loglik(x, comp) for comp in composite_lik.components]...)
end

"""
    _pointwise_loglik!(::ConditionallyIndependent, result, x, composite_lik::CompositeLikelihood) -> Vector{Float64}

In-place pointwise log-likelihood computation for composite likelihoods.

Fills `result` by writing each component's pointwise log-likelihoods to the appropriate
slice of the output vector. The `result` vector must have length equal to the total
number of observations across all components.
"""
function _pointwise_loglik!(::ConditionallyIndependent, result, x, composite_lik::CompositeLikelihood)
    # Check all components are conditionally independent
    for comp in composite_lik.components
        if observation_independence(comp) != ConditionallyIndependent()
            error(
                "CompositeLikelihood contains component with ConditionallyDependent trait.\n"
                    * "All components must have ConditionallyIndependent observations for pointwise_loglik!."
            )
        end
    end

    # Fill result by slicing for each component
    idx = 1
    for comp in composite_lik.components
        n_obs = length(comp.y)
        pointwise_loglik!(view(result, idx:(idx + n_obs - 1)), x, comp)
        idx += n_obs
    end

    return result
end
