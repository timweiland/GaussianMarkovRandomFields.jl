using Mooncake
using Mooncake: @is_primitive, @mooncake_overlay, MinimalCtx, CoDual, NoRData, NoFData, primal, tangent, fdata, zero_tangent
using SparseArrays: nonzeros, SparseMatrixCSC
using LinearAlgebra: Hermitian
using CliqueTrees.Multifrontal: ChordalCholesky

@is_primitive MinimalCtx Tuple{Type{ChordalGMRF}, AbstractVector, SparseMatrixCSC}

function Mooncake.rrule!!(
        ::CoDual{Type{ChordalGMRF}},
        cdμ::CoDual{<:AbstractVector},
        cdQ::CoDual{<:SparseMatrixCSC},
    )
    μ, Σμ = MooncakeSparse.primaltangent(cdμ)
    Q, ΣQ = MooncakeSparse.primaltangent(cdQ)

    gmrf = ChordalGMRF(μ, Q)
    dy = fdata(zero_tangent(gmrf))

    function pullback!!(::NoRData)
        dμ = MooncakeSparse.toarray(gmrf.μ, dy.data.μ)
        dQ = MooncakeSparse.toarray(gmrf.Q, dy.data.Q)

        Σμ .+= dμ
        nonzeros(ΣQ) .+= nonzeros(parent(dQ))

        return NoRData(), NoRData(), NoRData()
    end

    return CoDual(gmrf, dy), pullback!!
end

@is_primitive MinimalCtx Tuple{Type{ChordalGMRF}, AbstractVector, Hermitian, ChordalCholesky}

function Mooncake.rrule!!(
        ::CoDual{Type{ChordalGMRF}},
        cdμ::CoDual{<:AbstractVector},
        cdQ::CoDual{<:Hermitian},
        cdF::CoDual{<:ChordalCholesky},
    )
    μ, Σμ = MooncakeSparse.primaltangent(cdμ)
    Q, ΣQ = MooncakeSparse.primaltangent(cdQ)
    F = primal(cdF)

    gmrf = ChordalGMRF(μ, Q, F)
    dy = fdata(zero_tangent(gmrf))

    function pullback!!(::NoRData)
        dμ = MooncakeSparse.toarray(gmrf.μ, dy.data.μ)
        dQ = MooncakeSparse.toarray(gmrf.Q, dy.data.Q)

        Σμ .+= dμ
        nonzeros(parent(ΣQ)) .+= nonzeros(parent(dQ))

        return NoRData(), NoRData(), NoRData(), NoRData()
    end

    return CoDual(gmrf, dy), pullback!!
end

function gaussian_approximation_notangent(prior::ChordalGMRF, obslik::ObservationLikelihood; kwargs...)
    return gaussian_approximation(prior, obslik; kwargs...)
end

@is_primitive MinimalCtx Tuple{typeof(gaussian_approximation_notangent), ChordalGMRF, ObservationLikelihood}
@is_primitive MinimalCtx Tuple{typeof(Core.kwcall), Any, typeof(gaussian_approximation_notangent), ChordalGMRF, ObservationLikelihood}

function Mooncake.rrule!!(
        ::CoDual{typeof(gaussian_approximation_notangent)},
        cdprior::CoDual{<:ChordalGMRF},
        cdobslik::CoDual{<:ObservationLikelihood},
    )
    prior = primal(cdprior)
    obslik = primal(cdobslik)
    posterior = gaussian_approximation_notangent(prior, obslik)

    function pullback!!(::NoRData)
        return NoRData(), Mooncake.zero_rdata(prior), Mooncake.zero_rdata(obslik)
    end

    return CoDual(posterior, fdata(zero_tangent(posterior))), pullback!!
end

function Mooncake.rrule!!(
        ::CoDual{typeof(Core.kwcall)},
        cdkwargs::CoDual,
        ::CoDual{typeof(gaussian_approximation_notangent)},
        cdprior::CoDual{<:ChordalGMRF},
        cdobslik::CoDual{<:ObservationLikelihood},
    )
    prior = primal(cdprior)
    obslik = primal(cdobslik)
    kwargs = primal(cdkwargs)
    posterior = gaussian_approximation_notangent(prior, obslik; kwargs...)

    function pullback!!(::NoRData)
        return NoRData(), NoRData(), NoRData(), Mooncake.zero_rdata(prior), Mooncake.zero_rdata(obslik)
    end

    return CoDual(posterior, fdata(zero_tangent(posterior))), pullback!!
end

@mooncake_overlay function gaussian_approximation(
        prior::ChordalGMRF,
        obslik::ObservationLikelihood;
        kwargs...
    )
    posterior = gaussian_approximation_notangent(prior, obslik; kwargs...)
    x_star = mean(posterior)

    grad = ∇ₓ_neg_log_posterior(prior, obslik, x_star)
    x_corrected = x_star - posterior.F \ grad

    Q_post = hermdiff(precision_matrix(prior), loghessian(x_corrected, obslik))

    return ChordalGMRF(x_corrected, Q_post, posterior.F)
end
