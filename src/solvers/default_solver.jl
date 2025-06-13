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

function default_var_strategy(N::Integer)
    if N <= MC_THRESHOLD
        return TakahashiStrategy()
    else
        return RBMCStrategy(100)
    end
end

function construct_solver(
    sbp::DefaultSolverBlueprint,
    mean::AbstractVector,
    Q::LinearMaps.LinearMap # precision
)
    N = length(mean)
    var_strategy = default_var_strategy(N)
    if N <= CG_THRESHOLD
        return construct_solver(CholeskySolverBlueprint(var_strategy = var_strategy), mean, Q)
    else
        return construct_solver(CGSolverBlueprint(var_strategy = var_strategy), mean, Q)
    end
end

function construct_conditional_solver(
    sbp::DefaultSolverBlueprint,
    prior_mean::AbstractVector,
    Q::LinearMaps.LinearMap,
    A::LinearMaps.LinearMap,
    Q_ε::LinearMaps.LinearMap,
    y::AbstractVector,
    b::AbstractVector
    )
    N = length(prior_mean)
    var_strategy = default_var_strategy(N)

    if N <= CG_THRESHOLD
        return construct_conditional_solver(CholeskySolverBlueprint(var_strategy = var_strategy), prior_mean, Q, A, Q_ε, y, b)
    else
        return construct_conditional_solver(CGSolverBlueprint(var_strategy = var_strategy), prior_mean, Q, A, Q_ε, y, b)
    end
end
