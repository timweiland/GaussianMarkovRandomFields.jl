using IterativeSolvers

export GNLinearSolverBlueprint, GNCGSolverBlueprint, GNCholeskySolverBlueprint

abstract type GNLinearSolverBlueprint end

struct GNCholeskySolverBlueprint <: GNLinearSolverBlueprint
    perm::Union{Nothing,Vector{Int}}

    function GNCholeskySolverBlueprint(perm::Union{Nothing,Vector{Int}} = nothing)
        new(perm)
    end
end

struct GNCGSolverBlueprint <: GNLinearSolverBlueprint
    maxiter::Int
    reltol::Real
    abstol::Real
    preconditioner_fn::Function
    verbose::Bool

    function GNCGSolverBlueprint(
        ;
        maxiter::Int = 100,
        reltol::Real = 1e-6,
        abstol::Real = 1e-6,
        preconditioner_fn::Function = A -> IterativeSolvers.Identity(),
        verbose = false,
    )
        new(maxiter, reltol, abstol, preconditioner_fn, verbose)
    end
end
