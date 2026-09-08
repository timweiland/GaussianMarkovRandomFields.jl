# `ChordalGMRF`: constructor primitives that stop Mooncake at the factorization
# boundary. Its `logdetcov`/`var`/`logpdf` need no overlays — `MooncakeSparse`
# ships rules for the two-argument forms they reach.

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
