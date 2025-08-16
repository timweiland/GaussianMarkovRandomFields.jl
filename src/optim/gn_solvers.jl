using IterativeSolvers

export GNLinearSolverBlueprint, GNCGSolverBlueprint, GNCholeskySolverBlueprint

"""
    GNLinearSolverBlueprint

Abstract type for the specification of a solver for the linear systems arising
in Gauss-Newton optimization.
"""
abstract type GNLinearSolverBlueprint end

"""
    GNCholeskySolverBlueprint(perm)

Specification of a linear solver for Gauss-Newton systems based on the Cholesky
decomposition.
`perm` is a node reordering (*perm*utation) to minimize fill-in. If such a
reordering is available from previous computations, it can be reused here to
avoid unnecessary computational overhead. If `perm` is not passed, it will be
computed during the Cholesky decomposition.
"""
struct GNCholeskySolverBlueprint <: GNLinearSolverBlueprint
    perm::Union{Nothing, Vector{Int}}

    function GNCholeskySolverBlueprint(perm::Union{Nothing, Vector{Int}} = nothing)
        return new(perm)
    end
end

"""
    GNCGSolverBlueprint(; maxiter, reltol, abstol, preconditioner_fn, verbose)

Specification of a linear solver for Gauss-Newton systems based on the conjugate
gradient (CG) method.
"""
struct GNCGSolverBlueprint <: GNLinearSolverBlueprint
    maxiter::Int
    reltol::Real
    abstol::Real
    preconditioner_fn::Function
    verbose::Bool

    function GNCGSolverBlueprint(;
            maxiter::Int = 100,
            reltol::Real = 1.0e-6,
            abstol::Real = 1.0e-6,
            preconditioner_fn::Function = A -> IterativeSolvers.Identity(),
            verbose = false,
        )
        return new(maxiter, reltol, abstol, preconditioner_fn, verbose)
    end
end
