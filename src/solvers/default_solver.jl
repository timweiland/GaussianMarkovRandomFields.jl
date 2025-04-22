export DefaultSolverBlueprint, construct_solver

const CG_THRESHOLD = 250000 # Threshold for switching to CG-based solver
const MC_THRESHOLD = CG_THRESHOLD # Threshold for switching to Monte Carlo variance estimation

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

function default_var_strategy(gmrf::AbstractGMRF)
    N = length(gmrf)
    if N <= MC_THRESHOLD
        return TakahashiStrategy()
    else
        return RBMCStrategy(100)
    end
end

function construct_solver(::DefaultSolverBlueprint, gmrf::AbstractGMRF)
    N = length(gmrf)
    var_strategy = default_var_strategy(gmrf)
    if N <= CG_THRESHOLD
        return construct_solver(CholeskySolverBlueprint(var_strategy = var_strategy), gmrf)
    else
        return construct_solver(CGSolverBlueprint(var_strategy = var_strategy), gmrf)
    end
end
