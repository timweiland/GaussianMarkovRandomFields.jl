using CliqueTrees.Multifrontal: ChordalTriangular, Permutation, ChordalSymbolic, symbolic, chordal, selinv as mselinv, logdet
using LinearAlgebra: Hermitian, cholesky, diag, ldiv!, axpy!, dot
using SparseArrays: SparseMatrixCSC
using Random: AbstractRNG, randn

export ChordalGMRF

struct ChordalGMRF{T <: Real, Herm <: HermSparse{T}, Tri <: ChordalTriangular{:N, :L, T}, Prm <: Permutation, Mea <: AbstractVector{T}} <: AbstractGMRF{T, Herm}
    μ::Mea
    Q::Herm
    L::Tri
    P::Prm
end

function ChordalGMRF(μ::AbstractVector, Q::SparseMatrixCSC, L, P)
    return ChordalGMRF(μ, Hermitian(Q, :L), L, P)
end

function ChordalGMRF(μ::AbstractVector, Q::SparseMatrixCSC; kw...)
    H = Hermitian(Q, :L)
    P, S = symbolic(H; kw...)
    L = cholesky(chordal(H, P, S))
    return ChordalGMRF(μ, H, L, P)
end

function Base.length(d::ChordalGMRF)
    return length(d.μ)
end

function mean(d::ChordalGMRF)
    return d.μ
end

function precision_map(d::ChordalGMRF)
    return d.Q
end

function precision_matrix(d::ChordalGMRF)
    return d.Q
end

function logdetcov(d::ChordalGMRF)
    return -logdet(precision_matrix(d), d.L, d.P)
end

function sqmahal(d::ChordalGMRF, x::AbstractVector)
    r = x - d.μ
    return dot(r, precision_matrix(d), r)
end

function gradlogpdf(d::ChordalGMRF, x::AbstractVector)
    return precision_matrix(d) * (d.μ - x)
end

function var(d::ChordalGMRF)
    Σ = mselinv(precision_matrix(d), d.L, d.P)
    return diag(Σ)
end

function _rand!(rng::AbstractRNG, d::ChordalGMRF{T}, x::AbstractVector) where {T}
    z = randn(rng, T, length(x))
    return axpy!(1, d.μ, d.P \ ldiv!(d.L', d.P * z))
end

function Base.show(io::IO, d::ChordalGMRF{T}) where {T}
    return print(io, "ChordalGMRF{$T}(n=$(length(d)))")
end

function Base.show(io::IO, ::MIME"text/plain", d::ChordalGMRF{T}) where {T}
    println(io, "ChordalGMRF{$T} with $(length(d)) variables")

    μ = d.μ

    return if length(μ) <= 6
        print(io, "  Mean: $μ")
    else
        print(io, "  Mean: [$(μ[1]), $(μ[2]), $(μ[3]), ..., $(μ[end - 2]), $(μ[end - 1]), $(μ[end])]")
    end
end

# ChainRulesCore rrule for ChordalGMRF constructor
# ChordalGMRF is defined by (μ, Q). L and P are derived - gradients never flow through them.
using ChainRulesCore: ChainRulesCore, NoTangent

function ChainRulesCore.rrule(::Type{ChordalGMRF}, μ::AbstractVector, Q::SparseMatrixCSC; kw...)
    result = ChordalGMRF(μ, Q; kw...)
    ChordalGMRF_pullback(ȳ) = (NoTangent(), ȳ.μ, ȳ.Q)
    return result, ChordalGMRF_pullback
end
