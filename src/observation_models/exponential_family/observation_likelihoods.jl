export ExponentialFamilyLikelihood, NormalLikelihood, PoissonLikelihood, BernoulliLikelihood, BinomialLikelihood, NegBinLikelihood, GammaLikelihood, StudentTLikelihood

"""
    ExponentialFamilyLikelihood{L, I} <: ObservationLikelihood

Abstract type for exponential family observation likelihoods.

This intermediate type allows for generic implementations that work across all 
exponential family distributions while still allowing specialized methods for 
specific combinations.

# Type Parameters
- `L`: Link function type
- `I`: Index type (Nothing for non-indexed, UnitRange or Vector for indexed)
"""
abstract type ExponentialFamilyLikelihood{L, I} <: ObservationLikelihood end

"""
    NormalLikelihood{L<:LinkFunction} <: ObservationLikelihood

Materialized Normal observation likelihood with precomputed hyperparameters.

# Fields
- `link::L`: Link function connecting latent field to mean parameter
- `y::Vector{Float64}`: Observed data  
- `σ::Float64`: Standard deviation hyperparameter
- `inv_σ²::Float64`: Precomputed 1/σ² for performance
- `log_σ::Float64`: Precomputed log(σ) for log-likelihood computation

# Example
```julia
obs_model = ExponentialFamily(Normal)
obs_lik = obs_model([1.0, 2.0, 1.5]; σ=0.5)  # NormalLikelihood{IdentityLink}
ll = loglik([0.9, 2.1, 1.4], obs_lik)
```
"""
struct NormalLikelihood{L <: LinkFunction, I, T <: Real} <: ExponentialFamilyLikelihood{L, I}
    link::L
    y::Vector{Float64}
    σ::T
    inv_σ²::T
    log_σ::T
    indices::I  # Can be Nothing, UnitRange, or Vector{Int}
end

"""
    PoissonLikelihood{L<:LinkFunction, I, O} <: ObservationLikelihood

Materialized Poisson observation likelihood.

# Fields  
- `link::L`: Link function connecting latent field to rate parameter
- `y::Vector{Int}`: Count observations
- `indices::I`: Indices of the latent field corresponding to the observations
- `logexposure::O`: Log exposure / offset

# Example
```julia
obs_model = ExponentialFamily(Poisson)  # Uses LogLink by default
obs_lik = obs_model([1, 3, 0, 2])      # PoissonLikelihood{LogLink}
ll = loglik([0.0, 1.1, -2.0, 0.7], obs_lik)  # x values on log scale
```
"""
struct PoissonLikelihood{L <: LinkFunction, I, O} <: ExponentialFamilyLikelihood{L, I}
    link::L
    y::Vector{Int}
    indices::I  # Can be Nothing, UnitRange, or Vector{Int}
    logexposure::O   # Real vector
end

"""
    BernoulliLikelihood{L<:LinkFunction} <: ObservationLikelihood

Materialized Bernoulli observation likelihood for binary data.

# Fields
- `link::L`: Link function connecting latent field to probability parameter  
- `y::Vector{Int}`: Binary observations (0 or 1)

# Example
```julia
obs_model = ExponentialFamily(Bernoulli)  # Uses LogitLink by default
obs_lik = obs_model([1, 0, 1, 0])        # BernoulliLikelihood{LogitLink}
ll = loglik([0.5, -0.2, 1.1, -0.8], obs_lik)  # x values on logit scale
```
"""
struct BernoulliLikelihood{L <: LinkFunction, I} <: ExponentialFamilyLikelihood{L, I}
    link::L
    y::Vector{Int}
    indices::I  # Can be Nothing, UnitRange, or Vector{Int}
end

"""
    BinomialLikelihood{L<:LinkFunction} <: ObservationLikelihood

Materialized Binomial observation likelihood.

# Fields
- `link::L`: Link function connecting latent field to probability parameter
- `y::Vector{Int}`: Number of successes for each trial
- `n::Vector{Int}`: Number of trials per observation (can vary across observations)

# Example  
```julia
obs_model = ExponentialFamily(Binomial)  # Uses LogitLink by default
obs_lik = obs_model([3, 1, 4]; trials=[5, 8, 6])  # BinomialLikelihood{LogitLink}
ll = loglik([0.2, -1.0, 0.8], obs_lik)  # x values on logit scale
```
"""
struct BinomialLikelihood{L <: LinkFunction, I} <: ExponentialFamilyLikelihood{L, I}
    link::L
    y::Vector{Int}
    n::Vector{Int}  # Changed from Int to Vector{Int}
    indices::I  # Can be Nothing, UnitRange, or Vector{Int}
end

"""
    NegBinLikelihood{L<:LinkFunction, I, O} <: ExponentialFamilyLikelihood{L, I}

Materialized Negative Binomial (NB2) observation likelihood.

Uses the NB2 parameterization where Var(y) = μ + μ²/r. As r → ∞, this
converges to the Poisson distribution.

# Fields
- `link::L`: Link function connecting latent field to mean parameter
- `y::Vector{Int}`: Count observations
- `r::Float64`: Shape/size parameter (overdispersion; r > 0)
- `indices::I`: Indices of the latent field corresponding to the observations
- `logexposure::O`: Log exposure / offset

# Example
```julia
obs_model = ExponentialFamily(NegativeBinomial)
y = NegativeBinomialObservations([3, 1, 8])
obs_lik = obs_model(y; r=5.0)
ll = loglik([1.0, 0.5, 2.0], obs_lik)
```
"""
struct NegBinLikelihood{L <: LinkFunction, I, O, T <: Real} <: ExponentialFamilyLikelihood{L, I}
    link::L
    y::Vector{Int}
    r::T
    indices::I
    logexposure::O
end

"""
    GammaLikelihood{L<:LinkFunction, I} <: ExponentialFamilyLikelihood{L, I}

Materialized Gamma observation likelihood using the mean-shape parameterization.

Uses Var(y) = μ²/φ, where φ is the shape parameter. Larger φ means less dispersion.

# Fields
- `link::L`: Link function connecting latent field to mean parameter
- `y::Vector{Float64}`: Continuous positive observations
- `phi::Float64`: Shape parameter (φ > 0; controls precision)
- `indices::I`: Indices of the latent field corresponding to the observations

# Example
```julia
obs_model = ExponentialFamily(Gamma)
obs_lik = obs_model([1.5, 0.3, 4.2]; phi=3.0)
ll = loglik([0.4, -1.2, 1.4], obs_lik)
```
"""
struct GammaLikelihood{L <: LinkFunction, I, T <: Real} <: ExponentialFamilyLikelihood{L, I}
    link::L
    y::Vector{Float64}
    phi::T
    indices::I
end

"""
    StudentTLikelihood{L<:LinkFunction, I} <: ExponentialFamilyLikelihood{L, I}

Materialized Student-t observation likelihood using the unit-variance parameterization.

Uses the unit-variance parameterization where Var(y) = σ² for all ν > 2, achieved by
rescaling the standard t-distribution by √((ν−2)/ν). This makes σ directly comparable
to the Normal standard deviation — users can swap `Normal` for `TDist` and keep σ.

# Fields
- `link::L`: Link function connecting latent field to location parameter
- `y::Vector{Float64}`: Continuous observations
- `σ::Float64`: Scale parameter (σ > 0; same interpretation as Normal's σ)
- `ν::Float64`: Degrees of freedom (ν > 2; required for finite variance)
- `indices::I`: Indices of the latent field corresponding to the observations

# Example
```julia
obs_model = ExponentialFamily(TDist)
obs_lik = obs_model([1.5, -0.3, 4.2]; σ=2.0, ν=4.0)
ll = loglik([0.4, -1.2, 1.4], obs_lik)
```
"""
struct StudentTLikelihood{L <: LinkFunction, I, T <: Real} <: ExponentialFamilyLikelihood{L, I}
    link::L
    y::Vector{Float64}
    σ::T
    ν::T
    w::T       # Precomputed: σ²(ν−2)
    νp1::T     # Precomputed: ν+1
    σ_eff::T   # Precomputed: σ√((ν−2)/ν)
    indices::I
end
