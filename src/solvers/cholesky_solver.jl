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

mutable struct CholeskySolver{M, V<:AbstractVarianceStrategy, Tv<:Real, L, C} <: AbstractCholeskySolver{M, V}
    mean::Vector{Tv}
    precision::L
    precision_chol::C
    var_strategy::V
    perm::Union{Nothing,Vector{Int}}
    computed_var::Union{Nothing,Vector{Tv}}
    computed_logdetcov::Union{Nothing,Tv}

    function CholeskySolver{M}(
        mean::AbstractVector{Tv},
        precision::L,
        var_strategy::V,
        perm::Union{Nothing,Vector{Int}} = nothing,
    ) where {M, V<:AbstractVarianceStrategy, Tv<:Real, L<:Union{LinearMaps.LinearMap{Tv}, AbstractMatrix{Tv}}}
        precision_chol = linmap_cholesky(Val(M), precision; perm = perm)
        new{M, V, Tv, L, typeof(precision_chol)}(mean, precision, precision_chol, var_strategy, perm, nothing, nothing)
    end
end

mutable struct LinearConditionalCholeskySolver{M, V<:AbstractVarianceStrategy, Tv<:Real, L<:LinearMap{Tv}, C, LA, LN} <:
               AbstractCholeskySolver{M, V}
    prior_mean::Vector{Tv}
    precision::L
    precision_chol::C
    A::LA
    Q_ϵ::LN
    y::Vector{Tv}
    b::Vector{Tv}
    var_strategy::V
    perm::Union{Nothing,Vector{Int}}
    computed_posterior_mean::Union{Nothing,Vector{Tv}}
    computed_var::Union{Nothing,Vector{Tv}}
    computed_logdetcov::Union{Nothing,Tv}

    function LinearConditionalCholeskySolver{M}(
        prior_mean::AbstractVector{Tv},
        posterior_precision::LinearMaps.LinearMap,
        A::LinearMaps.LinearMap,
        Q_ϵ::LinearMaps.LinearMap,
        y::AbstractVector,
        b::AbstractVector,
        var_strategy::V,
        perm::Union{Nothing,Vector{Int}} = nothing,
    ) where {M, V<:AbstractVarianceStrategy, Tv<:Real}
        precision_chol = linmap_cholesky(Val(M), posterior_precision; perm = perm)
        new{M, V, Tv, typeof(posterior_precision), typeof(precision_chol), typeof(A), typeof(Q_ϵ)}(
            prior_mean,
            posterior_precision,
            precision_chol,
            A,
            Q_ϵ,
            y,
            b,
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
    CholeskySolverBlueprint{Method}(; var_strategy=TakahashiStrategy(), perm=nothing)
    CholeskySolverBlueprint(; var_strategy=TakahashiStrategy(), perm=nothing)

A blueprint for a direct solver that uses a sparse Cholesky decomposition.

# Type Parameters
- `Method::Symbol`: The factorization method, either `:default` or `:autodiffable`.
  - `:default` uses CHOLMOD (recommended for most cases)
  - `:autodiffable` uses LDLFactorizations.jl (required for automatic differentiation)

# Keyword Arguments
- `var_strategy::AbstractVarianceStrategy`: Strategy for computing the marginal
  variances of the GMRF. Defaults to `TakahashiStrategy()`.
- `perm::Union{Nothing,Vector{Int}}`: Permutation / node reordering to use for
  the Cholesky decomposition. Defaults to `nothing`, which means that the
  solver will compute its own permutation.

# Examples
```julia
# Default CHOLMOD-based solver
blueprint = CholeskySolverBlueprint()

# Autodiff-compatible solver
blueprint = CholeskySolverBlueprint{:autodiffable}()

# Custom variance strategy
blueprint = CholeskySolverBlueprint(var_strategy=RBMCStrategy(100))
```
"""
struct CholeskySolverBlueprint{M, V} <: AbstractSolverBlueprint
    var_strategy::V
    perm::Union{Nothing,Vector{Int}}
    function CholeskySolverBlueprint{M}(;
        var_strategy::V = TakahashiStrategy(),
        perm::Union{Nothing,Vector{Int}} = nothing,
    ) where {M, V<:AbstractVarianceStrategy}
        new{M, V}(var_strategy, perm)
    end

    function CholeskySolverBlueprint(;
        var_strategy::V = TakahashiStrategy(),
        perm::Union{Nothing,Vector{Int}} = nothing,
    ) where {V<:AbstractVarianceStrategy}
        new{:default, V}(var_strategy, perm)
    end
end

function construct_solver(
    bp::CholeskySolverBlueprint{M},
    mean::AbstractVector,
    Q::Union{LinearMaps.LinearMap, AbstractMatrix}
    ) where {M}
    return CholeskySolver{M}(mean, Q, bp.var_strategy, bp.perm)
end

function construct_conditional_solver(
    bp::CholeskySolverBlueprint{M},
    prior_mean::AbstractVector,
    posterior_precision::LinearMaps.LinearMap,
    A::LinearMaps.LinearMap,
    Q_ε::LinearMaps.LinearMap,
    y::AbstractVector,
    b::AbstractVector
    ) where {M}
    return LinearConditionalCholeskySolver{M}(
        prior_mean,
        posterior_precision,
        A,
        Q_ε,
        y,
        b,
        bp.var_strategy,
        bp.perm,
    )
end

function infer_solver_blueprint(
    s::Union{CholeskySolver{M}, LinearConditionalCholeskySolver{M}}
    ) where {M}
    return CholeskySolverBlueprint{M}(
        var_strategy=s.var_strategy,
        perm=s.perm,
    )
end
