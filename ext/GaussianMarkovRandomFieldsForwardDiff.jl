"""
    GaussianMarkovRandomFieldsForwardDiff

"""
module GaussianMarkovRandomFieldsForwardDiff

import GaussianMarkovRandomFields as GMRFs
import GaussianMarkovRandomFields: GMRF
import Distributions: logdetcov

using ForwardDiff
using LinearAlgebra
using LinearMaps
using LinearSolve

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

function _primal_precision(precision::LinearMaps.LinearMap)
    return ForwardDiff.value.(to_matrix(precision))
end

# Build the LinearSolve cache with primal data because caches cannot handle Dual numbers.
function _forwarddiff_cache(mean::AbstractVector, precision::PrecisionLike, alg)
    configured_alg = GMRFs.configure_algorithm(alg)
    primal_rhs = copy(_primal_mean(mean))
    primal_precision = _primal_precision(precision)
    prepared_precision = GMRFs.prepare_for_linsolve(primal_precision, configured_alg)
    prob = LinearProblem(prepared_precision, primal_rhs)
    return init(prob, configured_alg)
end

# Reuse the standard constructor while injecting the Dual-safe LinearSolve cache.
function _construct_forwarddiff_gmrf(mean::AbstractVector, precision::PrecisionLike, alg, Q_sqrt, rbmc_strategy, linsolve_cache)
    n = length(mean)
    n == size(precision, 1) == size(precision, 2) || throw(ArgumentError("size mismatch"))

    T = promote_type(eltype(mean), eltype(precision))
    mean_T = eltype(mean) === T ? mean : convert(AbstractVector{T}, mean)

    precision_T =
    if eltype(precision) === T
        precision
    elseif precision isa LinearMaps.LinearMap
        LinearMaps.LinearMap{T}(convert(AbstractMatrix{T}, to_matrix(precision)))
    else
        convert(AbstractMatrix{T}, precision)
    end

    cache = linsolve_cache === nothing ? _forwarddiff_cache(mean_T, precision_T, alg) : linsolve_cache

    return GMRFs.GMRF{T, typeof(mean_T), Nothing, typeof(precision_T), typeof(Q_sqrt), typeof(cache), typeof(rbmc_strategy)}(
        mean_T,
        nothing,
        precision_T,
        Q_sqrt,
        cache,
        rbmc_strategy,
    )
end

function GMRFs.GMRF(
        mean::AbstractVector{<:ForwardDiff.Dual},
        precision::AbstractMatrix,
        alg = nothing;
        Q_sqrt = nothing,
        rbmc_strategy = GMRFs.RBMCStrategy(1000),
        linsolve_cache = nothing,
    )
    return _construct_forwarddiff_gmrf(mean, precision, alg, Q_sqrt, rbmc_strategy, linsolve_cache)
end

function GMRFs.GMRF(
        mean::AbstractVector{<:ForwardDiff.Dual},
        precision::LinearMaps.LinearMap,
        alg = nothing;
        Q_sqrt = nothing,
        rbmc_strategy = GMRFs.RBMCStrategy(1000),
        linsolve_cache = nothing,
    )
    return _construct_forwarddiff_gmrf(mean, precision, alg, Q_sqrt, rbmc_strategy, linsolve_cache)
end

function GMRFs.GMRF(
        mean::AbstractVector,
        precision::AbstractMatrix{<:ForwardDiff.Dual},
        alg = nothing;
        Q_sqrt = nothing,
        rbmc_strategy = GMRFs.RBMCStrategy(1000),
        linsolve_cache = nothing,
    )
    return _construct_forwarddiff_gmrf(mean, precision, alg, Q_sqrt, rbmc_strategy, linsolve_cache)
end

function GMRFs.GMRF(
        mean::AbstractVector,
        precision::LinearMaps.LinearMap{<:ForwardDiff.Dual},
        alg = nothing;
        Q_sqrt = nothing,
        rbmc_strategy = GMRFs.RBMCStrategy(1000),
        linsolve_cache = nothing,
    )
    return _construct_forwarddiff_gmrf(mean, precision, alg, Q_sqrt, rbmc_strategy, linsolve_cache)
end

# NOTE: Combined Dual overloads remove ambiguities when both arguments carry Dual numbers.
function GMRFs.GMRF(
        mean::AbstractVector{<:ForwardDiff.Dual},
        precision::AbstractMatrix{<:ForwardDiff.Dual},
        alg = nothing;
        Q_sqrt = nothing,
        rbmc_strategy = GMRFs.RBMCStrategy(1000),
        linsolve_cache = nothing,
    )
    return _construct_forwarddiff_gmrf(mean, precision, alg, Q_sqrt, rbmc_strategy, linsolve_cache)
end

function GMRFs.GMRF(
        mean::AbstractVector{<:ForwardDiff.Dual},
        precision::LinearMaps.LinearMap{<:ForwardDiff.Dual},
        alg = nothing;
        Q_sqrt = nothing,
        rbmc_strategy = GMRFs.RBMCStrategy(1000),
        linsolve_cache = nothing,
    )
    return _construct_forwarddiff_gmrf(mean, precision, alg, Q_sqrt, rbmc_strategy, linsolve_cache)
end

function logdetcov(x::GMRF{<:ForwardDiff.Dual})
    Qinv = GMRFs.selinv(x.linsolve_cache)
    primal = GMRFs.logdet_cov(x.linsolve_cache)
    tangent = -dot(Qinv, x.precision)
    return ForwardDiff.Dual{ForwardDiff.tagtype(tangent)}(primal, ForwardDiff.partials(tangent)...)
end

end
