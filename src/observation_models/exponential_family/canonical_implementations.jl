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
    η = _eta(lik, x)
    μ = _mu(lik, η)
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
    η = _eta(lik, x)
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
function loggrad(x, lik::NormalLikelihood{IdentityLink})
    y = lik.y
    η = _eta(lik, x)
    g_obs = (y .- η) .* lik.inv_σ²
    return _embed_grad(lik, g_obs, length(x))
end

"""
    loggrad(x, lik::PoissonLikelihood{LogLink}) -> Vector{Float64}

Compute gradient of Poisson likelihood with canonical log link w.r.t. latent field x.
"""
function loggrad(x, lik::PoissonLikelihood{LogLink})
    y = lik.y
    η = _eta(lik, x)
    μ = _mu(lik, η)
    g_obs = y .- μ
    return _embed_grad(lik, g_obs, length(x))
end

"""
    loggrad(x, lik::BernoulliLikelihood{LogitLink}) -> Vector{Float64}

Compute gradient of Bernoulli likelihood with canonical logit link w.r.t. latent field x.
"""
function loggrad(x, lik::BernoulliLikelihood{LogitLink})
    y = lik.y
    η = _eta(lik, x)
    μ = logistic.(η)
    g_obs = y .- μ
    return _embed_grad(lik, g_obs, length(x))
end

"""
    loggrad(x, lik::BinomialLikelihood{LogitLink}) -> Vector{Float64}

Compute gradient of Binomial likelihood with canonical logit link w.r.t. latent field x.
"""
function loggrad(x, lik::BinomialLikelihood{LogitLink})
    y = lik.y
    n = lik.n
    η = _eta(lik, x)
    μ = logistic.(η)
    g_obs = y .- n .* μ
    return _embed_grad(lik, g_obs, length(x))
end

# ----------------------------- loghessian methods for canonical links --------------------------

"""
    loghessian(x, lik::NormalLikelihood{IdentityLink}) -> Diagonal{Float64}

Compute Hessian of Normal likelihood with canonical identity link w.r.t. latent field x.
"""
function loghessian(x, lik::NormalLikelihood{IdentityLink})
    d_obs = fill(-lik.inv_σ², length(lik.y))
    return _embed_diag(lik, d_obs, length(x))
end

"""
    loghessian(x, lik::PoissonLikelihood{LogLink}) -> Diagonal{Float64}

Compute Hessian of Poisson likelihood with canonical log link w.r.t. latent field x.
"""
function loghessian(x, lik::PoissonLikelihood{LogLink})
    η = _eta(lik, x)
    μ = _mu(lik, η)
    d_obs = -μ
    return _embed_diag(lik, d_obs, length(x))
end

"""
    loghessian(x, lik::BernoulliLikelihood{LogitLink}) -> Diagonal{Float64}

Compute Hessian of Bernoulli likelihood with canonical logit link w.r.t. latent field x.
"""
function loghessian(x, lik::BernoulliLikelihood{LogitLink})
    η = _eta(lik, x)
    μ = logistic.(η)
    d_obs = -μ .* (1 .- μ)
    return _embed_diag(lik, d_obs, length(x))
end

"""
    loghessian(x, lik::BinomialLikelihood{LogitLink}) -> Diagonal{Float64}

Compute Hessian of Binomial likelihood with canonical logit link w.r.t. latent field x.
"""
function loghessian(x, lik::BinomialLikelihood{LogitLink})
    n = lik.n
    η = _eta(lik, x)
    μ = logistic.(η)
    d_obs = -n .* μ .* (1 .- μ)
    return _embed_diag(lik, d_obs, length(x))
end
