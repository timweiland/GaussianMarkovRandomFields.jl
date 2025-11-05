using Distributions

# =======================================================================================
# POINTWISE LOG-LIKELIHOOD IMPLEMENTATIONS
# =======================================================================================

# Generic implementations for ExponentialFamilyLikelihood that work by calling logpdf
# element-wise instead of on the product_distribution

# ----------------------------- Generic pointwise_loglik --------------------------

"""
    _pointwise_loglik(::ConditionallyIndependent, x, lik::ExponentialFamilyLikelihood)

Generic pointwise log-likelihood implementation for exponential family likelihoods.

Computes per-observation log-likelihoods by applying logpdf element-wise instead of
using product_distribution.
"""
function _pointwise_loglik(::ConditionallyIndependent, x, lik::ExponentialFamilyLikelihood)
    η = _eta(lik, x)
    μ = _mu(lik, η)
    return _pointwise_logpdf(lik, μ)
end

"""
    _pointwise_loglik!(::ConditionallyIndependent, result, x, lik::ExponentialFamilyLikelihood)

Generic in-place pointwise log-likelihood for exponential family likelihoods.
"""
function _pointwise_loglik!(::ConditionallyIndependent, result, x, lik::ExponentialFamilyLikelihood)
    η = _eta(lik, x)
    μ = _mu(lik, η)
    _pointwise_logpdf!(result, lik, μ)
    return result
end

# ----------------------------- Family-specific pointwise logpdf --------------------------

# NormalLikelihood
function _pointwise_logpdf(lik::NormalLikelihood, μ)
    return logpdf.(Normal.(μ, lik.σ), lik.y)
end

function _pointwise_logpdf!(result, lik::NormalLikelihood, μ)
    @inbounds for i in eachindex(result, μ, lik.y)
        result[i] = logpdf(Normal(μ[i], lik.σ), lik.y[i])
    end
    return result
end

# PoissonLikelihood
function _pointwise_logpdf(lik::PoissonLikelihood, μ)
    return logpdf.(Poisson.(μ), lik.y)
end

function _pointwise_logpdf!(result, lik::PoissonLikelihood, μ)
    @inbounds for i in eachindex(result, μ, lik.y)
        result[i] = logpdf(Poisson(μ[i]), lik.y[i])
    end
    return result
end

# BernoulliLikelihood
function _pointwise_logpdf(lik::BernoulliLikelihood, μ)
    return logpdf.(Bernoulli.(μ), lik.y)
end

function _pointwise_logpdf!(result, lik::BernoulliLikelihood, μ)
    @inbounds for i in eachindex(result, μ, lik.y)
        result[i] = logpdf(Bernoulli(μ[i]), lik.y[i])
    end
    return result
end

# BinomialLikelihood
function _pointwise_logpdf(lik::BinomialLikelihood, μ)
    return logpdf.(Binomial.(lik.n, μ), lik.y)
end

function _pointwise_logpdf!(result, lik::BinomialLikelihood, μ)
    @inbounds for i in eachindex(result, μ, lik.y, lik.n)
        result[i] = logpdf(Binomial(lik.n[i], μ[i]), lik.y[i])
    end
    return result
end
