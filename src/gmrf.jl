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
using Memoize
using Random
using SparseArrays, SparseInverseSubset
using LinearMaps

export AbstractGMRF, GMRF, precision_map, precision_chol

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
    AbstractGMRF

A [Gaussian Markov Random Field](https://en.wikipedia.org/wiki/Markov_random_field#Gaussian) 
(GMRF) is a special case of a multivariate normal distribution where the precision matrix
is sparse. The zero entries in the precision correspond to conditional independencies.
"""
abstract type AbstractGMRF <: AbstractMvNormal end

length(::AbstractGMRF) = error("length not implemented for GMRF")
solver_ref(x::AbstractGMRF) = x.solver_ref
@memoize mean(s::AbstractGMRF) = compute_mean(solver_ref(s)[])
precision_map(::AbstractGMRF) = error("precision_mat not implemented for GMRF")

### Generic derived methods
invcov(d::AbstractGMRF) = Hermitian(sparse(precision_map(d)))
cov(d::AbstractGMRF) =
    precision_chol(d) \ Matrix{eltype(precision_map(d))}(I, size(precision_map(d))...)

# TODO: Throw this out at some point
@memoize precision_chol(d::AbstractGMRF) = cholesky(invcov(d))

logdetprecision(d::AbstractGMRF) = logdet(precision_chol(d))
logdetcov(d::AbstractGMRF) = -logdetprecision(d)

sqmahal(d::AbstractGMRF, x::AbstractVector) = (Δ = x - mean(d);
dot(Δ, precision_map(d) * Δ))
sqmahal!(r::AbstractVector, d::AbstractGMRF, x::AbstractVector) = (r .= sqmahal(d, x))

gradlogpdf(d::AbstractGMRF, x::AbstractVector) = -precision_map(d) * (x .- mean(d))

_rand!(rng::AbstractRNG, d::AbstractGMRF, x::AbstractVector) =
    compute_rand!(solver_ref(d)[], rng, x)

@memoize var(d::AbstractGMRF) = compute_variance(solver_ref(d)[])
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
        solver_blueprint::AbstractSolverBlueprint = CholeskySolverBlueprint(),
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
        solver_blueprint::AbstractSolverBlueprint = CholeskySolverBlueprint(),
    ) where {T} = GMRF(mean, LinearMap(precision), solver_blueprint)
end

length(d::GMRF) = length(d.mean)
mean(d::GMRF) = d.mean
precision_map(d::GMRF) = d.precision
