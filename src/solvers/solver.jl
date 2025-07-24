using Random

export gmrf_precision, compute_mean, compute_variance, compute_rand!
export compute_logdetcov
export infer_solver_blueprint, postprocess!

"""
    gmrf_precision(s::AbstractSolver)

Return the precision map of the GMRF associated with a solver.
"""
gmrf_precision(s::AbstractSolver) = s.precision

"""
    compute_mean(s::AbstractSolver)

Compute the mean of the GMRF associated with a solver.
"""
compute_mean(::AbstractSolver) = error("compute_mean not implemented for Solver")

"""
    compute_variance(s::AbstractSolver)

Compute the marginal variances of the GMRF associated with a solver.
"""
compute_variance(::AbstractSolver) = error("compute_variance not implemented for Solver")

"""
    compute_rand!(s::AbstractSolver, rng::Random.AbstractRNG, x::AbstractVector)

Generate a random sample from the GMRF associated with a solver.
"""
compute_rand!(::AbstractSolver, ::Random.AbstractRNG, ::AbstractVector) =
    error("compute_rand! not implemented for Solver")

"""
    compute_logdetcov(s::AbstractSolver)

Compute the log determinant of the covariance matrix of the GMRF associated with
a solver.
"""
compute_logdetcov(::AbstractSolver) = error("compute_logdetcov not implemented for Solver")

"""
    construct_solver(blueprint::AbstractSolverBlueprint, mean::AbstractVector, precision)

Construct a solver for a GMRF using a blueprint.

# Arguments
- `blueprint::AbstractSolverBlueprint`: The solver blueprint specifying the solver type and configuration.
- `mean::AbstractVector`: The mean vector of the GMRF.
- `precision`: The precision matrix (inverse covariance) of the GMRF. Can be a LinearMap or AbstractMatrix.

# Returns
A solver instance that can compute means, variances, samples, and other GMRF quantities.

# Notes
This is the primary interface for constructing solvers. The solver stores references to the mean
and precision, and provides methods like `compute_mean`, `compute_variance`, `compute_rand!`, etc.

For conditional GMRFs, use `construct_conditional_solver` instead.
"""
function construct_solver(
    sbp::AbstractSolverBlueprint,
    ::AbstractVector, # mean
    ::Union{LinearMaps.LinearMap, AbstractMatrix} # precision
)
    error("construct_solver not implemented for $(typeof(sbp))")
end

"""
    construct_conditional_solver(blueprint, prior_mean, posterior_precision, A, Q_ε, y, b)

Construct a solver for a conditional GMRF using a blueprint.

# Arguments
- `blueprint::AbstractSolverBlueprint`: The solver blueprint specifying the solver type and configuration.
- `prior_mean::AbstractVector`: The mean vector of the prior GMRF.
- `posterior_precision::LinearMaps.LinearMap`: The posterior precision matrix (Q_prior + A' * Q_ε * A).
- `A::LinearMaps.LinearMap`: The observation matrix.
- `Q_ε::LinearMaps.LinearMap`: The precision matrix of the observation noise.
- `y::AbstractVector`: The observation vector.
- `b::AbstractVector`: The offset vector.

# Returns
A conditional solver instance that can compute posterior means, variances, samples, etc.

# Notes
This interface is used internally by `LinearConditionalGMRF`. The posterior precision is 
pre-computed as `Q_prior + A' * Q_ε * A` to avoid recomputation in the solver.
"""
function construct_conditional_solver(
    sbp::AbstractSolverBlueprint,
    ::AbstractVector, # prior mean
    ::LinearMaps.LinearMap, # posterior precision
    ::LinearMaps.LinearMap, # A
    ::LinearMaps.LinearMap, # Q_ε
    ::AbstractVector, # y
    ::AbstractVector # b
    )
    error("construct_conditional_solver not implemented for $(typeof(sbp))")
end

"""
    infer_solver_blueprint(solver::AbstractSolver)

Infer a solver blueprint from a concrete solver instance.
"""
infer_solver_blueprint(::AbstractSolver) = error("Unable to infer blueprint for solver")

"""
    infer_solver_blueprint(x::AbstractGMRF)

Infer the solver blueprint of a GMRF.
"""
infer_solver_blueprint(x::AbstractGMRF) = infer_solver_blueprint(x.solver)

"""
    postprocess!(solver::AbstractSolver, gmrf::AbstractGMRF)

Postprocess a solver after GMRF construction is complete.
This allows the solver to optimize itself using full GMRF context,
such as setting up tailored preconditioners.

The default implementation is a no-op.
"""
postprocess!(::AbstractSolver, ::AbstractGMRF) = nothing
