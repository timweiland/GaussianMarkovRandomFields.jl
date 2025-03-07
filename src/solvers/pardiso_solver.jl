export PardisoGMRFSolverBlueprint

"""
    PardisoGMRFSolverBlueprint

A blueprint for a direct solver that uses Pardiso internally.

Highly efficient, but requires a Pardiso license.
"""
struct PardisoGMRFSolverBlueprint <: AbstractSolverBlueprint end

function construct_solver(_::PardisoGMRFSolverBlueprint, gmrf::AbstractGMRF)
    return error("Load Pardiso.jl to use this solver")
end

function construct_solver(_::PardisoGMRFSolverBlueprint, gmrf::LinearConditionalGMRF)
    return error("Load Pardiso.jl to use this solver")
end
