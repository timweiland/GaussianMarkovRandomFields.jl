using SparseArrays, LinearAlgebra, Distributions

export CholeskySolver, CholeskySolverBlueprint

_ensure_dense(x) = x
_ensure_dense(x::SparseVector) = Array(x)

struct CholeskySolver{G<:AbstractGMRF,V<:AbstractVarianceStrategy} <: AbstractSolver
    gmrf::G
    var_strategy::V
    precision_chol::Union{Cholesky,SparseArrays.CHOLMOD.Factor}

    function CholeskySolver(
        gmrf::G,
        var_strategy::V,
    ) where {G<:AbstractGMRF,V<:AbstractVarianceStrategy}
        precision_chol = cholesky(to_matrix(precision_map(gmrf)))
        new{G,V}(gmrf, var_strategy, precision_chol)
    end
end

function compute_mean(s::CholeskySolver)
    return s.gmrf.mean
end

function compute_mean(s::CholeskySolver{<:LinearConditionalGMRF})
    x = s.gmrf
    μ = _ensure_dense(mean(x.prior))
    residual = x.y - (x.A * μ + x.b)
    return μ + s.precision_chol \ _ensure_dense(x.A' * (x.Q_ϵ * residual))
end

function compute_variance(s::CholeskySolver)
    return compute_variance(s.var_strategy, s)
end

function compute_rand!(s::CholeskySolver, rng::Random.AbstractRNG, x::AbstractVector)
    randn!(rng, x)
    x .= s.precision_chol.UP \ _ensure_dense(x)
    x .+= _ensure_dense(mean(s.gmrf))
    return x
end

struct CholeskySolverBlueprint <: AbstractSolverBlueprint
    var_strategy::AbstractVarianceStrategy
    function CholeskySolverBlueprint(
        var_strategy::AbstractVarianceStrategy = RBMCStrategy(100),
    )
        new(var_strategy)
    end
end

function construct_solver(bp::CholeskySolverBlueprint, gmrf::AbstractGMRF)
    return CholeskySolver(gmrf, bp.var_strategy)
end
