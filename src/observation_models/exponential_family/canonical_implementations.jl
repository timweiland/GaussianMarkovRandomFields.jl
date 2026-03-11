using LinearAlgebra
using StatsFuns
using Distributions
using Distributions: product_distribution
using SpecialFunctions: loggamma

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

function _construct_distribution(lik::GammaLikelihood, μ)
    return product_distribution(Gamma.(lik.phi, μ ./ lik.phi))
end

function _construct_distribution(lik::StudentTLikelihood, μ)
    return product_distribution(μ .+ lik.σ_eff .* TDist(lik.ν))
end

# ----------------------------- Specialized loglik for NegBin --------------------------

"""
    loglik(x, lik::NegBinLikelihood) -> Float64

Fast implementation for Negative Binomial likelihood that avoids product_distribution overhead.

Computes: ∑ᵢ [loggamma(yᵢ+r) - loggamma(r) - loggamma(yᵢ+1) + r·log(r) + yᵢ·log(μᵢ) - (r+yᵢ)·log(r+μᵢ)]
"""
function loglik(x, lik::NegBinLikelihood)
    y = lik.y
    r = lik.r
    η = _eta(lik, x)
    μ = _mu(lik, η)

    ll = zero(eltype(μ))
    loggamma_r = loggamma(r)
    r_log_r = r * log(r)
    @inbounds for i in eachindex(y)
        ll += loggamma(y[i] + r) - loggamma_r - loggamma(y[i] + 1) +
            r_log_r + y[i] * log(μ[i]) - (r + y[i]) * log(r + μ[i])
    end
    return ll
end

# ----------------------------- Specialized loglik for Gamma --------------------------

"""
    loglik(x, lik::GammaLikelihood) -> Float64

Fast implementation for Gamma likelihood that avoids product_distribution overhead.

Computes: ∑ᵢ [φ log φ − φ log μᵢ − log Γ(φ) + (φ−1) log yᵢ − φyᵢ/μᵢ]
"""
function loglik(x, lik::GammaLikelihood)
    y = lik.y
    phi = lik.phi
    η = _eta(lik, x)
    μ = _mu(lik, η)

    n = length(y)
    phi_m1 = phi - 1
    ll = n * (phi * log(phi) - loggamma(phi))
    @inbounds for i in eachindex(y)
        ll += phi_m1 * log(y[i]) - phi * log(μ[i]) - phi * y[i] / μ[i]
    end
    return ll
end

# ----------------------------- Specialized loglik for Student-t --------------------------

"""
    loglik(x, lik::StudentTLikelihood) -> Float64

Fast implementation for Student-t likelihood using the unit-variance parameterization.

Computes: ∑ᵢ [loggamma((ν+1)/2) − loggamma(ν/2) − 0.5log(π(ν−2)) − log(σ)
               − (ν+1)/2 · log(1 + (yᵢ − μᵢ)² / (σ²(ν−2)))]
"""
function loglik(x, lik::StudentTLikelihood)
    y = lik.y
    w = lik.w
    η = _eta(lik, x)
    μ = _mu(lik, η)

    n = length(y)
    half_νp1 = lik.νp1 / 2
    ll = n * (loggamma(half_νp1) - loggamma(lik.ν / 2) - 0.5 * log(π * (lik.ν - 2)) - log(lik.σ))
    @inbounds for i in eachindex(y)
        ll -= half_νp1 * log(1 + (y[i] - μ[i])^2 / w)
    end
    return ll
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

"""
    loggrad(x, lik::NegBinLikelihood{LogLink}) -> Vector{Float64}

Compute gradient of Negative Binomial likelihood with log link w.r.t. latent field x.

∂ℓ/∂ηᵢ = r(yᵢ - μᵢ) / (r + μᵢ)
"""
function loggrad(x, lik::NegBinLikelihood{LogLink})
    y = lik.y
    r = lik.r
    η = _eta(lik, x)
    μ = _mu(lik, η)
    g_obs = @. r * (y - μ) / (r + μ)
    return _embed_grad(lik, g_obs, length(x))
end

"""
    loggrad(x, lik::GammaLikelihood{LogLink}) -> Vector{Float64}

Compute gradient of Gamma likelihood with log link w.r.t. latent field x.

∂ℓ/∂ηᵢ = φ(yᵢ/μᵢ − 1)
"""
function loggrad(x, lik::GammaLikelihood{LogLink})
    y = lik.y
    phi = lik.phi
    η = _eta(lik, x)
    μ = _mu(lik, η)
    g_obs = @. phi * (y / μ - 1)
    return _embed_grad(lik, g_obs, length(x))
end

"""
    loggrad(x, lik::StudentTLikelihood{IdentityLink}) -> Vector{Float64}

Compute gradient of Student-t likelihood with canonical identity link w.r.t. latent field x.

∂ℓ/∂ηᵢ = (ν+1)(yᵢ − ηᵢ) / (w + (yᵢ − ηᵢ)²)  where w = σ²(ν−2)
"""
function loggrad(x, lik::StudentTLikelihood{IdentityLink})
    y = lik.y
    w = lik.w
    νp1 = lik.νp1
    η = _eta(lik, x)
    g_obs = @. νp1 * (y - η) / (w + (y - η)^2)
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

"""
    loghessian(x, lik::NegBinLikelihood{LogLink}) -> Diagonal{Float64}

Compute Hessian of Negative Binomial likelihood with log link w.r.t. latent field x.

∂²ℓ/∂ηᵢ² = -rμᵢ(r + yᵢ) / (r + μᵢ)²
"""
function loghessian(x, lik::NegBinLikelihood{LogLink})
    y = lik.y
    r = lik.r
    η = _eta(lik, x)
    μ = _mu(lik, η)
    d_obs = @. -r * μ * (r + y) / (r + μ)^2
    return _embed_diag(lik, d_obs, length(x))
end

"""
    loghessian(x, lik::GammaLikelihood{LogLink}) -> Diagonal{Float64}

Compute Hessian of Gamma likelihood with log link w.r.t. latent field x.

∂²ℓ/∂ηᵢ² = −φyᵢ/μᵢ
"""
function loghessian(x, lik::GammaLikelihood{LogLink})
    y = lik.y
    phi = lik.phi
    η = _eta(lik, x)
    μ = _mu(lik, η)
    d_obs = @. -phi * y / μ
    return _embed_diag(lik, d_obs, length(x))
end

"""
    loghessian(x, lik::StudentTLikelihood{IdentityLink}) -> Diagonal{Float64}

Compute Hessian of Student-t likelihood with canonical identity link w.r.t. latent field x.

∂²ℓ/∂ηᵢ² = (ν+1)((yᵢ − ηᵢ)² − w) / (w + (yᵢ − ηᵢ)²)²  where w = σ²(ν−2)

Note: Can be positive when |y−η| > σ√(ν−2) — the Student-t log-likelihood is NOT globally concave.
"""
function loghessian(x, lik::StudentTLikelihood{IdentityLink})
    y = lik.y
    w = lik.w
    νp1 = lik.νp1
    η = _eta(lik, x)
    d_obs = @. νp1 * ((y - η)^2 - w) / (w + (y - η)^2)^2
    return _embed_diag(lik, d_obs, length(x))
end
