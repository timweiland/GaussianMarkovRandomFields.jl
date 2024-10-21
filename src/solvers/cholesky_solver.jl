using SparseArrays, LinearAlgebra, Distributions, LinearMaps

export AbstractCholeskySolver,
    CholeskySolver, LinearConditionalCholeskySolver, CholeskySolverBlueprint

_ensure_dense(x) = x
_ensure_dense(x::SparseVector) = Array(x)


abstract type AbstractCholeskySolver{V<:AbstractVarianceStrategy} <: AbstractSolver end

compute_mean(s::AbstractCholeskySolver) = s.mean
function compute_variance(s::AbstractCholeskySolver)
    if s.computed_var !== nothing
        return s.computed_var
    end
    s.computed_var = compute_variance(s.var_strategy, s)
    return s.computed_var
end
function compute_rand!(
    s::AbstractCholeskySolver,
    rng::Random.AbstractRNG,
    x::AbstractVector,
)
    randn!(rng, x)
    x .= s.precision_chol.UP \ _ensure_dense(x)
    x .+= _ensure_dense(compute_mean(s))
    return x
end

mutable struct CholeskySolver{V<:AbstractVarianceStrategy} <: AbstractCholeskySolver{V}
    mean::AbstractVector
    precision::LinearMap
    precision_chol::Union{Cholesky,SparseArrays.CHOLMOD.Factor}
    var_strategy::V
    perm::Union{Nothing,Vector{Int}}
    computed_var::Union{Nothing, AbstractVector}

    function CholeskySolver(
        gmrf::AbstractGMRF,
        var_strategy::V,
        perm::Union{Nothing,Vector{Int}} = nothing,
    ) where {V<:AbstractVarianceStrategy}
        mat = to_matrix(precision_map(gmrf))
        precision_chol = cholesky(mat; perm = perm)
        new{V}(mean(gmrf), precision_map(gmrf), precision_chol, var_strategy, perm, nothing)
    end
end

mutable struct LinearConditionalCholeskySolver{V<:AbstractVarianceStrategy} <:
       AbstractCholeskySolver{V}
    prior_mean::AbstractVector
    precision::LinearMap
    precision_chol::Union{Cholesky,SparseArrays.CHOLMOD.Factor}
    A::LinearMap
    Q_ϵ::LinearMap
    y::AbstractVector
    b::AbstractVector
    var_strategy::V
    perm::Union{Nothing,Vector{Int}}
    computed_posterior_mean::Union{Nothing, AbstractVector}
    computed_var::Union{Nothing, AbstractVector}

    function LinearConditionalCholeskySolver(
        gmrf::LinearConditionalGMRF,
        var_strategy::V,
        perm::Union{Nothing,Vector{Int}} = nothing,
    ) where {V<:AbstractVarianceStrategy}
        mat = to_matrix(precision_map(gmrf))
        precision_chol = cholesky(mat; perm = perm)
        new{V}(
            mean(gmrf.prior),
            precision_map(gmrf),
            precision_chol,
            gmrf.A,
            gmrf.Q_ϵ,
            gmrf.y,
            gmrf.b,
            var_strategy,
            perm,
            nothing,
            nothing,
        )
    end
end

function compute_mean(s::LinearConditionalCholeskySolver)
    if s.computed_posterior_mean !== nothing
        return s.computed_posterior_mean
    end
    μ = _ensure_dense(s.prior_mean)
    residual = _ensure_dense(s.y - (s.A * μ + s.b))
    rhs = _ensure_dense(s.A' * (s.Q_ϵ * residual))
    s.computed_posterior_mean = μ + s.precision_chol \ rhs
    return s.computed_posterior_mean
end

struct CholeskySolverBlueprint <: AbstractSolverBlueprint
    var_strategy::AbstractVarianceStrategy
    perm::Union{Nothing,Vector{Int}}
    function CholeskySolverBlueprint(
        ;
        var_strategy::AbstractVarianceStrategy = RBMCStrategy(100),
        perm::Union{Nothing,Vector{Int}} = nothing,
    )
        new(var_strategy, perm)
    end
end

function construct_solver(bp::CholeskySolverBlueprint, gmrf::AbstractGMRF)
    return CholeskySolver(gmrf, bp.var_strategy, bp.perm)
end

function construct_solver(bp::CholeskySolverBlueprint, gmrf::LinearConditionalGMRF)
    return LinearConditionalCholeskySolver(gmrf, bp.var_strategy, bp.perm)
end
