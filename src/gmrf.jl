import Base: length
import Distributions:
    AbstractMvNormal,
    mean,
    cov,
    invcov,
    logdetcov,
    sqmahal,
    sqmahal!,
    gradlogpdf,
    _rand!,
    var,
    std
using LinearAlgebra
using Random
using SparseArrays
using LinearMaps

export AbstractGMRF, GMRF, precision_map, precision_matrix

########################################################
#
#    AbstractGMRF
#
#    Abstract type for Gaussian Markov Random Fields
#    Each subtype must implement the following methods:
#    - length
#    - mean
#    - precision_map
#
########################################################
"""
    AbstractGMRF

A [Gaussian Markov Random Field](https://en.wikipedia.org/wiki/Markov_random_field#Gaussian) 
(GMRF) is a special case of a multivariate normal distribution where the precision matrix
is sparse. The zero entries in the precision correspond to conditional independencies.
"""
abstract type AbstractGMRF <: AbstractMvNormal end

solver_ref(x::AbstractGMRF) = x.solver_ref
mean(s::AbstractGMRF) = compute_mean(solver_ref(s)[])

"""
    precision_map(::AbstractGMRF)

Return the precision (inverse covariance) map of the GMRF.
"""
precision_map(::AbstractGMRF) = error("precision_map not implemented for GMRF")

"""
    precision_matrix(::AbstractGMRF)

Return the precision (inverse covariance) matrix of the GMRF.
"""
precision_matrix(x::AbstractGMRF) = to_matrix(precision_map(x))

length(d::AbstractGMRF) = Base.size(precision_map(d), 1)

### Generic derived methods
invcov(d::AbstractGMRF) = Hermitian(sparse(precision_map(d)))
cov(d::AbstractGMRF) =
    precision_chol(d) \ Matrix{eltype(precision_map(d))}(I, size(precision_map(d))...)

# TODO: Throw this out at some point
precision_chol(d::AbstractGMRF) = cholesky(invcov(d))

logdetprecision(d::AbstractGMRF) = logdet(precision_chol(d))
logdetcov(d::AbstractGMRF) = -logdetprecision(d)

sqmahal(d::AbstractGMRF, x::AbstractVector) = (Δ = x - mean(d);
dot(Δ, precision_map(d) * Δ))
sqmahal!(r::AbstractVector, d::AbstractGMRF, x::AbstractVector) = (r .= sqmahal(d, x))

gradlogpdf(d::AbstractGMRF, x::AbstractVector) = -precision_map(d) * (x .- mean(d))

_rand!(rng::AbstractRNG, d::AbstractGMRF, x::AbstractVector) =
    compute_rand!(solver_ref(d)[], rng, x)

var(d::AbstractGMRF) = compute_variance(solver_ref(d)[])
std(d::AbstractGMRF) = sqrt.(var(d))

#####################
#
#    GMRF
#
#####################
"""
    GMRF{T}(mean, precision, solver_ref)

A Gaussian Markov Random Field with mean `mean` and precision matrix `precision`.
Carries a reference to a solver for the GMRF quantities.
"""
struct GMRF{T} <: AbstractGMRF
    mean::AbstractVector{T}
    precision::LinearMap{T}
    solver_ref::Base.RefValue{AbstractSolver}

    function GMRF(
        mean::AbstractVector{T},
        precision::LinearMap{T},
        solver_blueprint::AbstractSolverBlueprint = DefaultSolverBlueprint(),
    ) where {T}
        n = length(mean)
        n == size(precision, 1) == size(precision, 2) ||
            throw(ArgumentError("size mismatch"))
        solver_ref = Base.RefValue{AbstractSolver}()
        self = new{T}(mean, precision, solver_ref)
        solver_ref[] = construct_solver(solver_blueprint, self)
        return self
    end

    GMRF(
        mean::AbstractVector{T},
        precision::AbstractMatrix{T},
        solver_blueprint::AbstractSolverBlueprint = DefaultSolverBlueprint(),
    ) where {T} = GMRF(mean, LinearMap(precision), solver_blueprint)
end

length(d::GMRF) = length(d.mean)
mean(d::GMRF) = d.mean
precision_map(d::GMRF) = d.precision
