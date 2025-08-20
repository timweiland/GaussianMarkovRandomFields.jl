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
