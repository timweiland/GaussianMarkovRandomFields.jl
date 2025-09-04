using LinearAlgebra
using StatsFuns
using Distributions
using Distributions: product_distribution

# =======================================================================================
# SPECIALIZED IMPLEMENTATIONS: Chain rule methods for non-canonical links
# =======================================================================================
#
# Note: These provide optimized implementations for specific exponential family types
# The general AD fallbacks are in base.jl

"""
    loggrad(x, lik::ExponentialFamilyLikelihood) -> Vector{Float64}

Optimized chain rule implementation for exponential family likelihoods.
Handles indexing via helper embedding.
"""
function loggrad(x, lik::ExponentialFamilyLikelihood)
    η = _eta(lik, x)
    μ = _mu(lik, η)
    dμ_dη = derivative_invlink.(Ref(lik.link), η)
    g_obs = _loggrad_family(lik, μ, dμ_dη)
    return _embed_grad(lik, g_obs, length(x))
end

"""
    loghessian(x, lik::ExponentialFamilyLikelihood) -> Diagonal{Float64}

Optimized chain rule implementation for exponential family likelihoods.
Handles indexing via helper embedding.
"""
function loghessian(x, lik::ExponentialFamilyLikelihood)
    η = _eta(lik, x)
    μ = _mu(lik, η)
    dμ_dη = derivative_invlink.(Ref(lik.link), η)
    d2μ_dη² = second_derivative_invlink.(Ref(lik.link), η)
    d_obs = _loghessian_diagonal_family(lik, μ, dμ_dη, d2μ_dη²)
    return _embed_diag(lik, d_obs, length(x))
end

# ----------------------------- Family-specific helper functions for gradients/hessians --------------------------

function _loggrad_family(lik::NormalLikelihood, μ, dμ_dη)
    y = lik.y
    return ((y .- μ) .* lik.inv_σ²) .* dμ_dη
end

function _loggrad_family(lik::PoissonLikelihood, μ, dμ_dη)
    y = lik.y
    return ((y .- μ) ./ μ) .* dμ_dη
end

function _loggrad_family(lik::BernoulliLikelihood, μ, dμ_dη)
    y = lik.y
    return ((y .- μ) ./ (μ .* (1 .- μ))) .* dμ_dη
end

function _loggrad_family(lik::BinomialLikelihood, μ, dμ_dη)
    y = lik.y
    n = lik.n
    return ((y .- n .* μ) ./ (μ .* (1 .- μ))) .* dμ_dη
end

function _loghessian_diagonal_family(lik::NormalLikelihood, μ, dμ_dη, d2μ_dη²)
    y = lik.y
    return -(dμ_dη .^ 2) .* lik.inv_σ² .+ (y .- μ) .* lik.inv_σ² .* d2μ_dη²
end

function _loghessian_diagonal_family(lik::PoissonLikelihood, μ, dμ_dη, d2μ_dη²)
    y = lik.y
    # ∂²ℓ/∂η² = (∂²ℓ/∂μ²) × (∂μ/∂η)² + (∂ℓ/∂μ) × (∂²μ/∂η²)
    # For Poisson: ∂²ℓ/∂μ² = -y/μ², ∂ℓ/∂μ = y/μ - 1
    d2l_dmu2 = -y ./ (μ .^ 2)
    dl_dmu = (y ./ μ) .- 1
    return d2l_dmu2 .* (dμ_dη .^ 2) .+ dl_dmu .* d2μ_dη²
end

function _loghessian_diagonal_family(lik::BernoulliLikelihood, μ, dμ_dη, d2μ_dη²)
    y = lik.y
    # ∂²ℓ/∂η² = (∂²ℓ/∂μ²) × (∂μ/∂η)² + (∂ℓ/∂μ) × (∂²μ/∂η²)
    # For Bernoulli: ∂²ℓ/∂μ² = -y/μ² - (1-y)/(1-μ)², ∂ℓ/∂μ = y/μ - (1-y)/(1-μ)
    d2l_dmu2 = -(y ./ (μ .^ 2)) .- ((1 .- y) ./ ((1 .- μ) .^ 2))
    dl_dmu = (y ./ μ) .- ((1 .- y) ./ (1 .- μ))
    return d2l_dmu2 .* (dμ_dη .^ 2) .+ dl_dmu .* d2μ_dη²
end

function _loghessian_diagonal_family(lik::BinomialLikelihood, μ, dμ_dη, d2μ_dη²)
    y = lik.y
    n = lik.n
    # ∂²ℓ/∂η² = (∂²ℓ/∂μ²) × (∂μ/∂η)² + (∂ℓ/∂μ) × (∂²μ/∂η²)
    # For Binomial: ∂²ℓ/∂μ² = -y/μ² - (n-y)/(1-μ)², ∂ℓ/∂μ = y/μ - (n-y)/(1-μ)
    d2l_dmu2 = -(y ./ (μ .^ 2)) .- ((n .- y) ./ ((1 .- μ) .^ 2))
    dl_dmu = (y ./ μ) .- ((n .- y) ./ (1 .- μ))
    return d2l_dmu2 .* (dμ_dη .^ 2) .+ dl_dmu .* d2μ_dη²
end
