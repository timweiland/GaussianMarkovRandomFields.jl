using SparseArrays, LinearAlgebra, Distributions, LinearMaps

export AbstractCholeskySolver,
    CholeskySolver, LinearConditionalCholeskySolver, CholeskySolverBlueprint

_ensure_dense(x) = x
_ensure_dense(x::SparseVector) = Array(x)


abstract type AbstractCholeskySolver{M, V<:AbstractVarianceStrategy} <: AbstractSolver end

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
    if s.precision_chol isa Cholesky
        x .= s.precision_chol.U \ _ensure_dense(x)
    else
        x .= s.precision_chol.UP \ _ensure_dense(x)
    end
    x .+= _ensure_dense(compute_mean(s))
    return x
end

function compute_logdetcov(
    s::AbstractCholeskySolver,
)
    if s.computed_logdetcov !== nothing
        return s.computed_logdetcov
    end
    s.computed_logdetcov = -logdet(s.precision_chol)
    return s.computed_logdetcov
end

mutable struct CholeskySolver{M, V<:AbstractVarianceStrategy} <: AbstractCholeskySolver{M, V}
    mean::AbstractVector
    precision::LinearMap
    precision_chol
    var_strategy::V
    perm::Union{Nothing,Vector{Int}}
    computed_var::Union{Nothing,AbstractVector}
    computed_logdetcov::Union{Nothing,Real}

    function CholeskySolver(
        gmrf::AbstractGMRF,
        var_strategy::V,
        perm::Union{Nothing,Vector{Int}} = nothing,
        factorization_method = :default,
    ) where {V<:AbstractVarianceStrategy}
        precision_chol = linmap_cholesky(precision_map(gmrf); perm = perm, method=factorization_method)
        new{factorization_method, V}(mean(gmrf), precision_map(gmrf), precision_chol, var_strategy, perm, nothing, nothing)
    end
end

mutable struct LinearConditionalCholeskySolver{M, V<:AbstractVarianceStrategy} <:
               AbstractCholeskySolver{M, V}
    prior_mean::AbstractVector
    precision::LinearMap
    precision_chol
    A::LinearMap
    Q_ϵ::LinearMap
    y::AbstractVector
    b::AbstractVector
    var_strategy::V
    perm::Union{Nothing,Vector{Int}}
    computed_posterior_mean::Union{Nothing,AbstractVector}
    computed_var::Union{Nothing,AbstractVector}
    computed_logdetcov::Union{Nothing,Real}

    function LinearConditionalCholeskySolver(
        gmrf::LinearConditionalGMRF,
        var_strategy::V,
        perm::Union{Nothing,Vector{Int}} = nothing,
        factorization_method = :default,
    ) where {V<:AbstractVarianceStrategy}
        precision_chol = linmap_cholesky(precision_map(gmrf); perm = perm, method=factorization_method)
        new{factorization_method, V}(
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

"""
    CholeskySolverBlueprint(;
        factorization_method = :default,
        var_strategy = RBMCStrategy(100),
        perm = nothing
        )

A blueprint for a direct solver that uses a sparse Cholesky decomposition
computed through CHOLMOD.

# Keyword arguments
- `factorization_method::Symbol`: One of [:default, :autodiffable].
  Uses CHOLMOD by default to factorize. If you need to autodiff through GMRF
  quantities, use :autodiffable, which switches to LDLFactorizations.jl.
- `var_strategy::AbstractVarianceStrategy`: Strategy for computing the marginal
   variances of the GMRF. Defaults to `RBMCStrategy(100)`.
- `perm::Union{Nothing,Vector{Int}}`: Permutation / node reordering to use for
    the Cholesky decomposition. Defaults to `nothing`, which means that the
    solver will compute its own permutation.
"""
struct CholeskySolverBlueprint <: AbstractSolverBlueprint
    factorization_method::Symbol
    var_strategy::AbstractVarianceStrategy
    perm::Union{Nothing,Vector{Int}}
    function CholeskySolverBlueprint(;
        factorization_method::Symbol = :default,
        var_strategy::AbstractVarianceStrategy = TakahashiStrategy(),
        perm::Union{Nothing,Vector{Int}} = nothing,
    )
        new(factorization_method, var_strategy, perm)
    end
end

function construct_solver(bp::CholeskySolverBlueprint, gmrf::AbstractGMRF)
    return CholeskySolver(gmrf, bp.var_strategy, bp.perm, bp.factorization_method)
end

function construct_solver(bp::CholeskySolverBlueprint, gmrf::LinearConditionalGMRF)
    return LinearConditionalCholeskySolver(gmrf, bp.var_strategy, bp.perm, bp.factorization_method)
end

function infer_solver_blueprint(
    s::Union{CholeskySolver{M}, LinearConditionalCholeskySolver{M}}
    ) where {M}
    return CholeskySolverBlueprint(
        factorization_method=M,
        var_strategy=s.var_strategy,
        perm=s.perm,
    )
end
