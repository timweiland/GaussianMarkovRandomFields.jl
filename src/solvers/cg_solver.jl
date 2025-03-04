using SparseArrays, LinearAlgebra, Distributions, IterativeSolvers, LinearMaps

export AbstractCGSolver,
    CGSolver, LinearConditionalCGSolver, CGSolverBlueprint, construct_solver, compute_mean

import Base: step

abstract type AbstractCGSolver{V<:AbstractVarianceStrategy} <: AbstractSolver end

compute_mean(s::AbstractCGSolver) = s.mean
function compute_variance(s::AbstractCGSolver)
    if s.computed_var !== nothing
        return s.computed_var
    end
    s.computed_var = compute_variance(s.var_strategy, s)
    return s.computed_var
end
function compute_rand!(s::AbstractCGSolver, rng::Random.AbstractRNG, x::AbstractVector)
    if s.Q_sqrt === nothing
        s.Q_sqrt = linmap_sqrt(s.precision)
    end
    w = s.Q_sqrt * randn(rng, Base.size(s.Q_sqrt, 2))
    x .= zeros(Base.size(w))
    cg!(
        x,
        s.precision,
        w;
        maxiter = s.maxiter,
        reltol = s.reltol,
        abstol = s.abstol,
        verbose = true,
        Pl = s.Pl,
    )
    x .+= compute_mean(s)
    return x
end

mutable struct CGSolver{V<:AbstractVarianceStrategy} <: AbstractCGSolver{V}
    mean::AbstractVector
    precision::LinearMap
    reltol::Real
    abstol::Real
    maxiter::Int
    Pl::Union{Identity,AbstractPreconditioner}
    Q_sqrt::Union{Nothing,LinearMaps.LinearMap}
    var_strategy::V
    computed_var::Union{Nothing,AbstractVector}

    function CGSolver(
        gmrf::AbstractGMRF,
        reltol::Real,
        abstol::Real,
        maxiter::Int,
        Pl::Union{Identity,AbstractPreconditioner},
        var_strategy::V,
    ) where {V<:AbstractVarianceStrategy}
        new{V}(
            mean(gmrf),
            precision_map(gmrf),
            reltol,
            abstol,
            maxiter,
            Pl,
            nothing,
            var_strategy,
            nothing,
        )
    end
end

mutable struct LinearConditionalCGSolver{V<:AbstractVarianceStrategy} <: AbstractCGSolver{V}
    prior_mean::AbstractVector
    precision::LinearMap
    A::LinearMap
    Q_ϵ::LinearMap
    y::AbstractVector
    b::AbstractVector
    reltol::Real
    abstol::Real
    maxiter::Int
    Pl::Union{Identity,AbstractPreconditioner}
    var_strategy::V
    Q_sqrt::Union{Nothing,LinearMaps.LinearMap}
    mean_residual_guess::Union{Nothing,AbstractVector}
    computed_posterior_mean::Union{Nothing,AbstractVector}
    computed_var::Union{Nothing,AbstractVector}

    function LinearConditionalCGSolver(
        gmrf::LinearConditionalGMRF,
        reltol::Real,
        abstol::Real,
        maxiter::Int,
        Pl::Union{Identity,AbstractPreconditioner},
        var_strategy::V,
        mean_residual_guess::Union{Nothing,AbstractVector},
    ) where {V<:AbstractVarianceStrategy}
        new{V}(
            mean(gmrf.prior),
            precision_map(gmrf),
            gmrf.A,
            gmrf.Q_ϵ,
            gmrf.y,
            gmrf.b,
            reltol,
            abstol,
            maxiter,
            Pl,
            var_strategy,
            nothing,
            mean_residual_guess,
            nothing,
            nothing,
        )
    end
end

function compute_mean(s::LinearConditionalCGSolver)
    if s.computed_posterior_mean !== nothing
        return s.computed_posterior_mean
    end
    residual = s.y - (s.A * s.prior_mean + s.b)
    rhs = s.A' * (s.Q_ϵ * residual)
    mean_residual =
        s.mean_residual_guess === nothing ? zeros(Base.size(rhs)) :
        copy(s.mean_residual_guess)
    cg!(
        mean_residual,
        s.precision,
        Array(rhs);
        maxiter = s.maxiter,
        reltol = s.reltol,
        abstol = s.abstol,
        verbose = true,
        Pl = s.Pl,
    )
    s.computed_posterior_mean = s.prior_mean + mean_residual
    return s.computed_posterior_mean
end

function default_preconditioner_strategy(::AbstractGMRF)
    return Identity()
end

"""
    CGSolverBlueprint(; reltol, abstol, maxiter, preconditioner_strategy,
                        var_strategy, mean_residual_guess)

A blueprint for a conjugate gradient-based solver.

# Keyword arguments
- `reltol::Real = sqrt(eps(Float64))`: Relative tolerance of CG.
- `abstol::Real = 0.0`: Absolute tolerance of CG.
- `maxiter::Int = 1000`: Maximum number of iterations.
- `preconditioner_strategy::Function`: Maps a GMRF instance to a preconditioner.
- `var_strategy::AbstractVarianceStrategy`: A variance strategy.
- `mean_residual_guess::Union{Nothing,AbstractVector}`: An initial guess for the
                                                        mean residual.
"""
struct CGSolverBlueprint <: AbstractSolverBlueprint
    reltol::Real
    abstol::Real
    maxiter::Int
    preconditioner_strategy::Function
    var_strategy::AbstractVarianceStrategy
    mean_residual_guess::Union{Nothing,AbstractVector}

    function CGSolverBlueprint(;
        reltol::Real = sqrt(eps(Float64)),
        abstol::Real = 0.0,
        maxiter::Int = 1000,
        preconditioner_strategy::Function = default_preconditioner_strategy,
        var_strategy::AbstractVarianceStrategy = RBMCStrategy(100),
        mean_residual_guess::Union{Nothing,AbstractVector} = nothing,
    )
        new(
            reltol,
            abstol,
            maxiter,
            preconditioner_strategy,
            var_strategy,
            mean_residual_guess,
        )
    end
end

function construct_solver(bp::CGSolverBlueprint, gmrf::AbstractGMRF)
    Pl = bp.preconditioner_strategy(gmrf)
    return CGSolver(gmrf, bp.reltol, bp.abstol, bp.maxiter, Pl, bp.var_strategy)
end

function construct_solver(bp::CGSolverBlueprint, gmrf::LinearConditionalGMRF)
    Pl = bp.preconditioner_strategy(gmrf)
    return LinearConditionalCGSolver(
        gmrf,
        bp.reltol,
        bp.abstol,
        bp.maxiter,
        Pl,
        bp.var_strategy,
        bp.mean_residual_guess,
    )
end
