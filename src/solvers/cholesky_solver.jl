using SparseArrays, LinearAlgebra, Distributions

export CholeskySolver, CholeskySolverBlueprint

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
    residual = x.y - (x.A * x.prior.mean + x.b)
    return mean(x.prior) + s.precision_chol \ (x.A' * (x.Q_ϵ * residual))
end

function compute_variance(s::CholeskySolver)
    return compute_variance(s.var_strategy, s)
end

function compute_rand!(s::CholeskySolver, rng::Random.AbstractRNG, x::AbstractVector)
    randn!(rng, x)
    x .= s.precision_chol.UP \ x
    x .+= mean(s.gmrf)
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
