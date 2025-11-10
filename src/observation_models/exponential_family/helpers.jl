using LinearAlgebra

# Small, inlined helpers to centralize indexing and offset handling.

@inline _eta(::ExponentialFamilyLikelihood{L, Nothing}, x) where {L} = x
@inline _eta(lik::ExponentialFamilyLikelihood{L, I}, x) where {L, I} = view(x, lik.indices)

# Default: apply inverse link directly
@inline function _mu(lik::ExponentialFamilyLikelihood, η)
    return apply_invlink.(Ref(lik.link), η)
end

# Specializations: Poisson with LogLink supports additive offsets on η
@inline _mu(lik::PoissonLikelihood{LogLink}, η) = exp.(η .+ lik.logexposure)

# Embed observation-space gradient back into full latent space when indexed
@inline _embed_grad(::ExponentialFamilyLikelihood{L, Nothing}, g_obs, n_full::Integer) where {L} = g_obs
@inline function _embed_grad(lik::ExponentialFamilyLikelihood{L, I}, g_obs, n_full::Integer) where {L, I}
    g = zeros(eltype(g_obs), n_full)
    g[lik.indices] .= g_obs
    return g
end

# Embed diagonal (observation-space) back into full latent space as a Diagonal
@inline _embed_diag(::ExponentialFamilyLikelihood{L, Nothing}, d_obs, n_full::Integer) where {L} = Diagonal(d_obs)
@inline function _embed_diag(lik::ExponentialFamilyLikelihood{L, I}, d_obs, n_full::Integer) where {L, I}
    d_full = zeros(eltype(d_obs), n_full)
    d_full[lik.indices] .= d_obs
    return Diagonal(d_full)
end
