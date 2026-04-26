# Shared helpers and type aliases used across the ForwardDiff extension.

const PrecisionLike = Union{LinearMaps.LinearMap, AbstractMatrix}

function _primal_mean(mean::AbstractVector)
    return ForwardDiff.value.(mean)
end

function _primal_precision(precision::AbstractMatrix)
    return ForwardDiff.value.(precision)
end

function _primal_precision(precision::SymTridiagonal)
    return SymTridiagonal(ForwardDiff.value.(precision.dv), ForwardDiff.value.(precision.ev))
end

function _primal_precision(precision::Diagonal)
    return Diagonal(ForwardDiff.value.(precision.diag))
end

function _primal_precision(precision::Symmetric{<:ForwardDiff.Dual, <:SparseMatrixCSC})
    return Symmetric(ForwardDiff.value.(precision.data))
end

function _primal_precision(precision::LinearMaps.LinearMap)
    return ForwardDiff.value.(to_matrix(precision))
end

# Extract primal values from observation likelihoods that may contain Dual hyperparameters.
# Default: identity (Poisson, Bernoulli, Binomial have no Real-typed hyperparameters)
_primal_obs_lik(obs_lik) = obs_lik

function _primal_obs_lik(lik::GMRFs.NormalLikelihood)
    return GMRFs.NormalLikelihood(
        lik.link, lik.y, ForwardDiff.value(lik.σ),
        ForwardDiff.value(lik.inv_σ²), ForwardDiff.value(lik.log_σ), lik.indices
    )
end

function _primal_obs_lik(lik::GMRFs.NegBinLikelihood)
    return GMRFs.NegBinLikelihood(
        lik.link, lik.y, ForwardDiff.value(lik.r), lik.indices, lik.logexposure
    )
end

function _primal_obs_lik(lik::GMRFs.GammaLikelihood)
    return GMRFs.GammaLikelihood(lik.link, lik.y, ForwardDiff.value(lik.phi), lik.indices)
end

function _primal_obs_lik(lik::GMRFs.StudentTLikelihood)
    return GMRFs.StudentTLikelihood(
        lik.link, lik.y, ForwardDiff.value(lik.σ), ForwardDiff.value(lik.ν),
        ForwardDiff.value(lik.w), ForwardDiff.value(lik.νp1),
        ForwardDiff.value(lik.σ_eff), lik.indices
    )
end

# Detect Dual-valued observation likelihoods via their type parameter T
const _DualNormalLik = GMRFs.NormalLikelihood{<:GMRFs.LinkFunction, <:Any, <:ForwardDiff.Dual}
const _DualNegBinLik = GMRFs.NegBinLikelihood{<:GMRFs.LinkFunction, <:Any, <:Any, <:ForwardDiff.Dual}
const _DualGammaLik = GMRFs.GammaLikelihood{<:GMRFs.LinkFunction, <:Any, <:ForwardDiff.Dual}
const _DualStudentTLik = GMRFs.StudentTLikelihood{<:GMRFs.LinkFunction, <:Any, <:ForwardDiff.Dual}
const _DualObsLik = Union{_DualNormalLik, _DualNegBinLik, _DualGammaLik, _DualStudentTLik}

function _dual_type_from_obs_lik(::GMRFs.NormalLikelihood{L, I, D}) where {L, I, D}
    return D
end
function _dual_type_from_obs_lik(::GMRFs.NegBinLikelihood{L, I, O, D}) where {L, I, O, D}
    return D
end
function _dual_type_from_obs_lik(::GMRFs.GammaLikelihood{L, I, D}) where {L, I, D}
    return D
end
function _dual_type_from_obs_lik(::GMRFs.StudentTLikelihood{L, I, D}) where {L, I, D}
    return D
end
