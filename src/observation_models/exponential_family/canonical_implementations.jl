using LinearAlgebra
using StatsFuns
using Distributions
using Distributions: product_distribution

# =======================================================================================
# EXPONENTIAL FAMILY IMPLEMENTATIONS: Generic loglik + specialized gradients/hessians
# =======================================================================================

# ----------------------------- Generic loglik using product_distribution --------------------------

"""
    loglik(x, lik::ExponentialFamilyLikelihood) -> Float64

Generic loglik implementation for all exponential family likelihoods using product_distribution.
"""
function loglik(x, lik::ExponentialFamilyLikelihood)
    y = lik.y
    # Use indexed view if indices are specified, otherwise use full x
    η = lik.indices === nothing ? x : view(x, lik.indices)
    μ = apply_invlink.(Ref(lik.link), η)
    dist = _construct_distribution(lik, μ)
    return logpdf(dist, y)
end

"""
    loglik(x, lik::NormalLikelihood) -> Float64

Specialized fast implementation for Normal likelihood that avoids product_distribution overhead.

Computes: ∑ᵢ logpdf(Normal(μᵢ, σ), yᵢ) = -n/2 * log(2π) - n * log(σ) - 1/(2σ²) * ∑ᵢ(yᵢ - μᵢ)²
"""
function loglik(x, lik::NormalLikelihood)
    y = lik.y
    # Use indexed view if indices are specified, otherwise use full x
    η = lik.indices === nothing ? x : view(x, lik.indices)
    μ = apply_invlink.(Ref(lik.link), η)

    # Fast computation avoiding product_distribution
    n = length(y)
    residuals = y .- μ
    sum_sq_residuals = sum(abs2, residuals)

    # -n/2 * log(2π) - n * log(σ) - 1/(2σ²) * ∑(yᵢ - μᵢ)²
    return -0.5 * n * log(2π) - n * lik.log_σ - 0.5 * lik.inv_σ² * sum_sq_residuals
end

# Family-specific distribution construction
function _construct_distribution(lik::NormalLikelihood, μ)
    return product_distribution(Normal.(μ, lik.σ))
end

function _construct_distribution(lik::PoissonLikelihood, μ)
    return product_distribution(Poisson.(μ))
end

function _construct_distribution(lik::BernoulliLikelihood, μ)
    return product_distribution(Bernoulli.(μ))
end

function _construct_distribution(lik::BinomialLikelihood, μ)
    return product_distribution(Binomial.(lik.n, μ))
end

# ----------------------------- loggrad methods for canonical links --------------------------

"""
    loggrad(x, lik::NormalLikelihood{IdentityLink}) -> Vector{Float64}

Compute gradient of Normal likelihood with canonical identity link w.r.t. latent field x.
"""
# Non-indexed case (indices === nothing)
function loggrad(x, lik::NormalLikelihood{IdentityLink, Nothing})
    y = lik.y
    μ = x  # Canonical identity link: μ = x
    return (y .- μ) .* lik.inv_σ²
end

# Indexed case (indices !== nothing)
function loggrad(x, lik::NormalLikelihood{IdentityLink})
    y = lik.y
    grad = zeros(eltype(x), length(x))
    μ = view(x, lik.indices)
    grad[lik.indices] .= (y .- μ) .* lik.inv_σ²
    return grad
end

"""
    loggrad(x, lik::PoissonLikelihood{LogLink}) -> Vector{Float64}

Compute gradient of Poisson likelihood with canonical log link w.r.t. latent field x.
"""
# Non-indexed case
function loggrad(x, lik::PoissonLikelihood{LogLink, Nothing})
    y = lik.y
    η = x
    μ = exp.(η)  # Canonical log link: μ = exp(η)
    return y .- μ
end

# Indexed case
function loggrad(x, lik::PoissonLikelihood{LogLink})
    y = lik.y
    grad = zeros(eltype(x), length(x))
    η = view(x, lik.indices)
    μ = exp.(η)
    grad[lik.indices] .= y .- μ
    return grad
end

"""
    loggrad(x, lik::BernoulliLikelihood{LogitLink}) -> Vector{Float64}

Compute gradient of Bernoulli likelihood with canonical logit link w.r.t. latent field x.
"""
# Non-indexed case
function loggrad(x, lik::BernoulliLikelihood{LogitLink, Nothing})
    y = lik.y
    η = x
    μ = logistic.(η)  # Canonical logit link: μ = logistic(η)
    return y .- μ
end

# Indexed case
function loggrad(x, lik::BernoulliLikelihood{LogitLink})
    y = lik.y
    grad = zeros(eltype(x), length(x))
    η = view(x, lik.indices)
    μ = logistic.(η)
    grad[lik.indices] .= y .- μ
    return grad
end

"""
    loggrad(x, lik::BinomialLikelihood{LogitLink}) -> Vector{Float64}

Compute gradient of Binomial likelihood with canonical logit link w.r.t. latent field x.
"""
# Non-indexed case
function loggrad(x, lik::BinomialLikelihood{LogitLink, Nothing})
    y = lik.y
    n = lik.n
    η = x
    μ = logistic.(η)  # Canonical logit link: μ = logistic(η)
    return y .- n .* μ
end

# Indexed case
function loggrad(x, lik::BinomialLikelihood{LogitLink})
    y = lik.y
    n = lik.n
    grad = zeros(eltype(x), length(x))
    η = view(x, lik.indices)
    μ = logistic.(η)
    grad[lik.indices] .= y .- n .* μ
    return grad
end

# ----------------------------- loghessian methods for canonical links --------------------------

"""
    loghessian(x, lik::NormalLikelihood{IdentityLink}) -> Diagonal{Float64}

Compute Hessian of Normal likelihood with canonical identity link w.r.t. latent field x.
"""
# Non-indexed case
function loghessian(x, lik::NormalLikelihood{IdentityLink, Nothing})
    return Diagonal(-ones(length(x)) .* lik.inv_σ²)
end

# Indexed case
function loghessian(x, lik::NormalLikelihood{IdentityLink})
    diagonal_terms = zeros(eltype(x), length(x))
    diagonal_terms[lik.indices] .= -lik.inv_σ²
    return Diagonal(diagonal_terms)
end

"""
    loghessian(x, lik::PoissonLikelihood{LogLink}) -> Diagonal{Float64}

Compute Hessian of Poisson likelihood with canonical log link w.r.t. latent field x.
"""
# Non-indexed case
function loghessian(x, lik::PoissonLikelihood{LogLink, Nothing})
    η = x
    μ = exp.(η)  # Canonical log link: μ = exp(η)
    return Diagonal(-μ)
end

# Indexed case
function loghessian(x, lik::PoissonLikelihood{LogLink})
    diagonal_terms = zeros(eltype(x), length(x))
    η = view(x, lik.indices)
    μ = exp.(η)
    diagonal_terms[lik.indices] .= -μ
    return Diagonal(diagonal_terms)
end

"""
    loghessian(x, lik::BernoulliLikelihood{LogitLink}) -> Diagonal{Float64}

Compute Hessian of Bernoulli likelihood with canonical logit link w.r.t. latent field x.
"""
# Non-indexed case
function loghessian(x, lik::BernoulliLikelihood{LogitLink, Nothing})
    η = x
    μ = logistic.(η)  # Canonical logit link: μ = logistic(η)
    return Diagonal(-μ .* (1 .- μ))
end

# Indexed case
function loghessian(x, lik::BernoulliLikelihood{LogitLink})
    diagonal_terms = zeros(eltype(x), length(x))
    η = view(x, lik.indices)
    μ = logistic.(η)
    diagonal_terms[lik.indices] .= -μ .* (1 .- μ)
    return Diagonal(diagonal_terms)
end

"""
    loghessian(x, lik::BinomialLikelihood{LogitLink}) -> Diagonal{Float64}

Compute Hessian of Binomial likelihood with canonical logit link w.r.t. latent field x.
"""
# Non-indexed case
function loghessian(x, lik::BinomialLikelihood{LogitLink, Nothing})
    n = lik.n
    η = x
    μ = logistic.(η)  # Canonical logit link: μ = logistic(η)
    return Diagonal(-n .* μ .* (1 .- μ))
end

# Indexed case
function loghessian(x, lik::BinomialLikelihood{LogitLink})
    n = lik.n
    diagonal_terms = zeros(eltype(x), length(x))
    η = view(x, lik.indices)
    μ = logistic.(η)
    diagonal_terms[lik.indices] .= -n .* μ .* (1 .- μ)
    return Diagonal(diagonal_terms)
end
