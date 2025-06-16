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
abstract type AbstractGMRF{T<:Real, L<:LinearMaps.LinearMap{T}} <: AbstractMvNormal end

solver(x::AbstractGMRF) = x.solver
mean(s::AbstractGMRF) = compute_mean(solver(s))

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
invcov(d::AbstractGMRF) = Symmetric(precision_matrix(d))
cov(::AbstractGMRF) = error("Prevented forming dense covariance matrix in memory.")

logdetcov(d::AbstractGMRF) = compute_logdetcov(solver(d))

sqmahal(d::AbstractGMRF, x::AbstractVector) = (Δ = x - mean(d);
dot(Δ, precision_map(d) * Δ))
sqmahal!(r::AbstractVector, d::AbstractGMRF, x::AbstractVector) = (r .= sqmahal(d, x))

gradlogpdf(d::AbstractGMRF, x::AbstractVector) = -precision_map(d) * (x .- mean(d))

_rand!(rng::AbstractRNG, d::AbstractGMRF, x::AbstractVector) =
    compute_rand!(solver(d), rng, x)

var(d::AbstractGMRF) = compute_variance(solver(d))
std(d::AbstractGMRF) = sqrt.(var(d))

#####################
#
#    GMRF
#
#####################
"""
    GMRF(mean, precision, solver_blueprint=DefaultSolverBlueprint())

A Gaussian Markov Random Field with mean `mean` and precision matrix `precision`.

# Arguments
- `mean::AbstractVector`: The mean vector of the GMRF.
- `precision::LinearMap`: The precision matrix (inverse covariance) of the GMRF.
- `solver_blueprint::AbstractSolverBlueprint`: Blueprint specifying how to construct the solver.

# Type Parameters
- `T<:Real`: The numeric type (e.g., Float64).
- `PrecisionMap<:LinearMap{T}`: The type of the precision matrix LinearMap.
- `Solver<:AbstractSolver`: The type of the solver instance.

# Fields
- `mean::Vector{T}`: The mean vector.
- `precision::PrecisionMap`: The precision matrix as a LinearMap.
- `solver::Solver`: The solver instance for computing GMRF quantities.

# Notes
The solver is constructed automatically using `construct_solver(solver_blueprint, mean, precision)`
and is used to compute means, variances, samples, and other GMRF quantities efficiently.
"""
struct GMRF{T<:Real, PrecisionMap<:LinearMap{T}, Solver<:AbstractSolver} <: AbstractGMRF{T, PrecisionMap}
    mean::Vector{T}
    precision::PrecisionMap
    solver::Solver
    #solver_ref::Base.RefValue{AbstractSolver}

    function GMRF(
        mean::AbstractVector,
        precision::PrecisionMap,
        solver_blueprint::AbstractSolverBlueprint = DefaultSolverBlueprint(),
    ) where {PrecisionMap <: LinearMap}
        n = length(mean)
        n == size(precision, 1) == size(precision, 2) ||
            throw(ArgumentError("size mismatch"))
        T = promote_type(eltype(mean), eltype(precision))
        if eltype(mean) != T
            mean = convert(AbstractVector{T}, mean)
        end
        if eltype(precision) != T
            precision = LinearMap{T}(convert(AbstractMatrix{T}, to_matrix(precision)))
        end

        #solver_ref = Base.RefValue{AbstractSolver}()
        solver = construct_solver(solver_blueprint, mean, precision)
        result = new{T, typeof(precision), typeof(solver)}(mean, precision, solver)
        postprocess!(solver, result)
        return result
        #solver_ref[] = construct_solver(solver_blueprint, self)
        #return self
    end

    GMRF(
        mean::AbstractVector,
        precision::AbstractMatrix,
        solver_blueprint::AbstractSolverBlueprint = DefaultSolverBlueprint(),
    ) = GMRF(mean, LinearMap(precision), solver_blueprint)
end

length(d::GMRF) = length(d.mean)
mean(d::GMRF) = d.mean
precision_map(d::GMRF) = d.precision
