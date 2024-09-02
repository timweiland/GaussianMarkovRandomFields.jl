export DefaultSolverBlueprint, construct_solver

"""
    DefaultSolverBlueprint()

Default solver blueprint which switches from Cholesky to CG based on
the size of the GMRF.
"""
struct DefaultSolverBlueprint <: AbstractSolverBlueprint
    function DefaultSolverBlueprint()
        new()
    end
end

function construct_solver(::DefaultSolverBlueprint, gmrf::AbstractGMRF)
    N = size(precision_map(gmrf), 1)
    if N < 100000
        return construct_solver(CholeskySolverBlueprint(), gmrf)
    else
        return construct_solver(CGSolverBlueprint(), gmrf)
    end
end
