using CliqueTrees.Multifrontal: ChordalCholesky, selinv as mselinv, logdet
using LinearAlgebra: Hermitian, cholesky!, diag, ldiv!, axpy!, dot
using SparseArrays: SparseMatrixCSC
using Random: AbstractRNG, randn

export ChordalGMRF

"""
    ChordalGMRF{T, Hrm, Fac, Mea} <: AbstractGMRF{T, Hrm}

A `GMRF` backed by a chordal Cholesky factorization (via
`CliqueTrees.Multifrontal.ChordalCholesky`) instead of CHOLMOD.

The pure-Julia chordal factorization composes naturally with `Mooncake`'s
reverse-mode AD through the rrules shipped by `MooncakeSparse`, so `logpdf`
and `gaussian_approximation` give correct gradients with respect to the
hyperparameters that produced `Q`. This is the recommended GMRF type for
Mooncake-based hyperparameter optimization (e.g. L-BFGS / Adam on the
marginal likelihood).

# Fields
- `μ::AbstractVector`: Mean.
- `Q::Hermitian`: Precision matrix.
- `F::ChordalCholesky`: Chordal Cholesky factorization of `Q`.

# Construction
```julia
ChordalGMRF(μ, Q)              # factorize Q via ChordalCholesky
ChordalGMRF(μ, Q, F)           # reuse a precomputed factorization
```
"""
struct ChordalGMRF{T <: Real, Hrm <: Hermitian, Fac <: ChordalCholesky, Mea <: AbstractVector{T}} <: AbstractGMRF{T, Hrm}
    μ::Mea
    Q::Hrm
    F::Fac
end

function ChordalGMRF(μ::AbstractVector, Q::SparseMatrixCSC, F::ChordalCholesky)
    return ChordalGMRF(μ, Hermitian(Q, :L), F)
end

function ChordalGMRF(μ::AbstractVector, Q::SparseMatrixCSC; kw...)
    H = Hermitian(Q, :L)
    F = cholesky!(ChordalCholesky(H; kw...))
    return ChordalGMRF(μ, H, F)
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
    return -logdet(d.Q, d.F)
end

function sqmahal(d::ChordalGMRF, x::AbstractVector)
    r = x - d.μ
    return dot(r, d.Q, r)
end

function gradlogpdf(d::ChordalGMRF, x::AbstractVector)
    return d.Q * (d.μ - x)
end

function var(d::ChordalGMRF)
    Σ = mselinv(d.Q, d.F)
    return diag(Σ)
end

function _rand!(rng::AbstractRNG, d::ChordalGMRF{T}, x::AbstractVector) where {T}
    z = randn(rng, T, length(x))
    return axpy!(true, d.μ, d.F.P \ ldiv!(d.F.U, z))
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
