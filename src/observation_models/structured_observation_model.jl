export StructuredObservationModel, ObsFactorGroup

"""
    ObsFactorGroup{K, F}

A group of observation factors sharing one log-likelihood *form*. Factor `i` couples the `K`
latent variables `vars[i]` to observation `y[obs_idx[i]]`, with log-likelihood
`loglik(vals, yk, θ::NamedTuple)` where `vals = x[vars[i]]`. AD specialises on the one small
`K`-input function per group, not on the whole-data likelihood.
"""
struct ObsFactorGroup{K, F}
    vars::Vector{NTuple{K, Int}}
    obs_idx::Vector{Int}
    loglik::F
end

"""
    StructuredObservationModel{G, HN} <: ObservationModel

Observation model expressed as a factor graph: a tuple of [`ObsFactorGroup`](@ref)s whose
per-observation log-likelihoods sum to `log p(y | x, θ)`. The gradient and (sparse) Hessian in
`x` are assembled by differentiating each group's small factor function and scattering — the
structured counterpart of [`AutoDiffObservationModel`](@ref). Works for any observation family
(`logpdf(Dist(params(x, θ)), yk)`), not just Gaussian.
"""
struct StructuredObservationModel{G <: Tuple, HN} <: ObservationModel
    n_latent::Int
    groups::G
    hyperparams::HN
end

StructuredObservationModel(n_latent::Int, groups::Tuple; hyperparams = ()) =
    StructuredObservationModel(n_latent, groups, Tuple(hyperparams))

struct StructuredObservationLikelihood{G <: Tuple, Y, Θ} <: ObservationLikelihood
    n_latent::Int
    groups::G
    y::Y
    θ::Θ
end

(m::StructuredObservationModel)(y; kwargs...) =
    StructuredObservationLikelihood(m.n_latent, m.groups, y, NamedTuple(kwargs))
latent_dimension(m::StructuredObservationModel, y) = m.n_latent
hyperparameters(m::StructuredObservationModel) = m.hyperparams

# --- type-stable recursion over the heterogeneous tuple of obs groups ---
@inline _obs_value(::Tuple{}, x, y, θ, s) = s
@inline function _obs_value(groups::Tuple, x, y, θ, s)
    s = _obs_value_one(first(groups), x, y, θ, s)
    return _obs_value(Base.tail(groups), x, y, θ, s)
end
function _obs_value_one(grp::ObsFactorGroup{K}, x, y, θ, s) where {K}
    f = grp.loglik
    @inbounds for i in eachindex(grp.vars)
        vars = grp.vars[i]
        vals = [x[vars[k]] for k in 1:K]
        s += f(vals, y[grp.obs_idx[i]], θ)
    end
    return s
end

@inline _obs_grad(::Tuple{}, x, y, θ, g) = g
@inline function _obs_grad(groups::Tuple, x, y, θ, g)
    _obs_grad_one(first(groups), x, y, θ, g)
    return _obs_grad(Base.tail(groups), x, y, θ, g)
end
function _obs_grad_one(grp::ObsFactorGroup{K}, x, y, θ, g) where {K}
    @inbounds for i in eachindex(grp.vars)
        vars = grp.vars[i]
        yk = y[grp.obs_idx[i]]
        f = v -> grp.loglik(v, yk, θ)
        vals = [x[vars[k]] for k in 1:K]
        gl = DI.gradient(f, DI.AutoForwardDiff(), vals)
        for li in 1:K
            g[vars[li]] += gl[li]
        end
    end
    return g
end

@inline _obs_hess(::Tuple{}, x, y, θ, Is, Js, Vs) = nothing
@inline function _obs_hess(groups::Tuple, x, y, θ, Is, Js, Vs)
    _obs_hess_one(first(groups), x, y, θ, Is, Js, Vs)
    return _obs_hess(Base.tail(groups), x, y, θ, Is, Js, Vs)
end
function _obs_hess_one(grp::ObsFactorGroup{K}, x, y, θ, Is, Js, Vs) where {K}
    @inbounds for i in eachindex(grp.vars)
        vars = grp.vars[i]
        yk = y[grp.obs_idx[i]]
        f = v -> grp.loglik(v, yk, θ)
        vals = [x[vars[k]] for k in 1:K]
        Hl = DI.hessian(f, DI.AutoForwardDiff(), vals)
        for li in 1:K, lj in 1:K
            push!(Is, vars[li])
            push!(Js, vars[lj])
            push!(Vs, Hl[li, lj])
        end
    end
    return nothing
end

function loglik(x, lik::StructuredObservationLikelihood)
    T = _structured_eltype(x, lik.θ)
    return _obs_value(lik.groups, _as_eltype(T, x), lik.y, lik.θ, zero(T))
end

function loggrad(x, lik::StructuredObservationLikelihood)
    T = _structured_eltype(x, lik.θ)
    g = zeros(T, lik.n_latent)
    return _obs_grad(lik.groups, _as_eltype(T, x), lik.y, lik.θ, g)
end

function loghessian(x, lik::StructuredObservationLikelihood)
    T = _structured_eltype(x, lik.θ)
    Is = Int[]
    Js = Int[]
    Vs = T[]
    _obs_hess(lik.groups, _as_eltype(T, x), lik.y, lik.θ, Is, Js, Vs)
    return sparse(Is, Js, Vs, lik.n_latent, lik.n_latent)
end

function pointwise_loglik(x, lik::StructuredObservationLikelihood)
    out = Float64[]
    for grp in lik.groups
        for i in eachindex(grp.vars)
            vars = grp.vars[i]
            vals = [x[vars[k]] for k in 1:length(vars)]
            push!(out, grp.loglik(vals, lik.y[grp.obs_idx[i]], lik.θ))
        end
    end
    return out
end
