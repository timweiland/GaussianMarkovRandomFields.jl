import Base: length
import Distributions: AbstractMvNormal, mean, cov, invcov,
    logdetcov, sqmahal, sqmahal!, gradlogpdf, _rand!
using LinearAlgebra
using Memoize
using Random

export AbstractGMRF, GMRF, precision_mat, precision_chol

########################################################
#
#    AbstractGMRF
#
#    Abstract type for Gaussian Markov Random Fields
#    Each subtype must implement the following methods:
#    - length
#    - mean
#    - precision_mat
#
########################################################
"""
A [Gaussian Markov Random Field](https://en.wikipedia.org/wiki/Markov_random_field#Gaussian) 
(GMRF) is a special case of a multivariate normal distribution where the precision matrix
is sparse. The zero entries in the precision correspond to conditional independencies.
"""
abstract type AbstractGMRF <: AbstractMvNormal end

length(::AbstractGMRF) = error("length not implemented for GMRF")
mean(::AbstractGMRF) = error("mean not implemented for GMRF")
precision_mat(::AbstractGMRF) = error("precision_mat not implemented for GMRF")

### Generic derived methods
invcov(d::AbstractGMRF) = precision_mat(d)
cov(d::AbstractGMRF) = precision_chol(d) \ Matrix{eltype(precision_mat(d))}(I, size(precision_mat(d))...)

@memoize precision_chol(d::AbstractGMRF) = cholesky(precision_mat(d))

logdetprecision(d::AbstractGMRF) = logdet(precision_chol(d))
logdetcov(d::AbstractGMRF) = -logdetprecision(d)

sqmahal(d::AbstractGMRF, x::AbstractVector) = (Δ = x - mean(d);
dot(Δ, precision_mat(d) * Δ))
sqmahal!(r::AbstractVector, d::AbstractGMRF, x::AbstractVector) = (r .= sqmahal(d, x))

gradlogpdf(d::AbstractGMRF, x::AbstractVector) = -precision_mat(d) * (x .- mean(d))

function _rand!(rng::AbstractRNG, d::AbstractGMRF, x::AbstractVector)
    randn!(rng, x)
    x .= precision_chol(d).UP \ x
    x .+= mean(d)
    return x
end

#####################
#
#    GMRF
#
#####################
"""
    GMRF(mean::Vector, precision::Matrix)

Construct a Gaussian Markov Random Field with mean `mean` and precision matrix `precision`.
"""
struct GMRF{T} <: AbstractGMRF
    mean::AbstractVector{T}
    precision::AbstractMatrix{T}

    function GMRF(mean::AbstractVector{T}, precision::AbstractMatrix{T}) where {T}
        n = length(mean)
        n == size(precision, 1) == size(precision, 2) || throw(ArgumentError("size mismatch"))
        new{T}(mean, precision)
    end
end

length(d::GMRF) = length(d.mean)
mean(d::GMRF) = d.mean
precision_mat(d::GMRF) = d.precision
