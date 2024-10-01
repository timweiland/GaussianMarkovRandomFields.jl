using SparseArrays, LinearAlgebra, Distributions, IterativeSolvers, LinearMaps

export CGSolver, CGSolverBlueprint, construct_solver, compute_mean

import Base: step

struct CGSolver{G<:AbstractGMRF} <: AbstractSolver
    gmrf::G
    reltol::Real
    abstol::Real
    maxiter::Int
    Pl::Union{Identity,AbstractPreconditioner}
    Q_sqrt::LinearMaps.LinearMap

    function CGSolver(
        gmrf::G,
        reltol::Real,
        abstol::Real,
        maxiter::Int,
        Pl::Union{Identity,AbstractPreconditioner},
    ) where {G}
        Q_sqrt = linmap_sqrt(gmrf.precision)
        new{G}(gmrf, reltol, abstol, maxiter, Pl, Q_sqrt)
    end
end

function compute_mean(s::CGSolver)
    return s.gmrf.mean
end

function compute_mean(s::CGSolver{<:LinearConditionalGMRF})
    x = s.gmrf
    residual = x.y - (x.A * x.prior.mean + x.b)
    rhs = x.A' * (x.Q_ϵ * residual)
    return mean(x.prior) + cg(
        x.precision,
        Array(rhs);
        maxiter = s.maxiter,
        reltol = s.reltol,
        abstol = s.abstol,
        verbose = true,
        Pl = s.Pl,
    )
end

function compute_rand!(s::CGSolver, rng::Random.AbstractRNG, x::AbstractVector)
    w = s.Q_sqrt * randn(rng, Base.size(s.Q_sqrt, 2))
    x .= zeros(size(w))
    cg!(
        x,
        s.gmrf.precision,
        w;
        maxiter = s.maxiter,
        reltol = s.reltol,
        abstol = s.abstol,
        verbose = true,
        Pl = s.Pl,
    )
    x .+= mean(s.gmrf)
    return x
end


function default_preconditioner_strategy(::AbstractGMRF)
    return Identity()
end

function default_preconditioner_strategy(
    x::Union{<:ConstantMeshSTGMRF,LinearConditionalGMRF{<:ConstantMeshSTGMRF}},
)
    block_size = N_spatial(x)
    Q = sparse(to_matrix(precision_map(x)))
    return temporal_block_gauss_seidel(Q, block_size)
end

struct CGSolverBlueprint <: AbstractSolverBlueprint
    reltol::Real
    abstol::Real
    maxiter::Int
    preconditioner_strategy::Function

    function CGSolverBlueprint(
        reltol::Real = sqrt(eps(Float64)),
        abstol::Real = 0.0,
        maxiter::Int = 1000,
        preconditioner_strategy::Function = default_preconditioner_strategy,
    )
        new(reltol, abstol, maxiter, preconditioner_strategy)
    end
end

function construct_solver(bp::CGSolverBlueprint, gmrf::AbstractGMRF)
    Pl = bp.preconditioner_strategy(gmrf)
    return CGSolver(gmrf, bp.reltol, bp.abstol, bp.maxiter, Pl)
end
