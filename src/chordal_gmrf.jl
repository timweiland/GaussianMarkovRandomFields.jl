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
hyperparameters that produced `Q`.

The same factorization is also available as a standard `GMRF` backend via
`GMRF(μ, Q, LinearSolve.CliqueTreesFactorization())`, which offers the same
Mooncake support plus the full LinearSolve-based feature set (information
vector constructors, conjugate conditioning, RBMC fallbacks). Prefer the
backend form unless you specifically want to manage the `ChordalCholesky`
factorization yourself.

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

# `AbstractGMRF{T, L}` requires `L <: AbstractMatrix{T}`, so the mean and the
# precision must share an element type. Promoting here is what lets a Float64
# mean be combined with a Dual precision (the shape ForwardDiff produces when
# only hyperparameters carry partials); without it the struct's type bound
# rejects the pair outright.
function _chordal_promote(μ::AbstractVector, Q::SparseMatrixCSC)
    T = promote_type(eltype(μ), eltype(Q))
    μ_T = eltype(μ) === T ? μ : convert(AbstractVector{T}, μ)
    # Rebuild rather than `convert` so the stored sparsity pattern is preserved
    # exactly, including any structural zeros.
    Q_T = eltype(Q) === T ? Q :
        SparseMatrixCSC(Q.m, Q.n, Q.colptr, Q.rowval, convert(Vector{T}, Q.nzval))
    return μ_T, Q_T
end

function ChordalGMRF(μ::AbstractVector, Q::SparseMatrixCSC, F::ChordalCholesky)
    μ_T, Q_T = _chordal_promote(μ, Q)
    return ChordalGMRF(μ_T, Hermitian(Q_T, :L), F)
end

function ChordalGMRF(μ::AbstractVector, Q::SparseMatrixCSC; kw...)
    μ_T, Q_T = _chordal_promote(μ, Q)
    H = Hermitian(Q_T, :L)
    F = cholesky!(ChordalCholesky(H; kw...))
    return ChordalGMRF(μ_T, H, F)
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
    # `diag` of a sparse matrix is a `SparseVector`, but every marginal variance
    # is stored anyway and a sparse result breaks `ForwardDiff.jacobian`.
    return Vector(diag(Σ))
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
