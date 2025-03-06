using Random

export gmrf_precision, compute_mean, compute_variance, compute_rand!

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
    construct_solver(blueprint::AbstractSolverBlueprint, gmrf::AbstractGMRF)

Construct a solver for a GMRF using a blueprint.
May be used to provide specialized solvers for specific GMRF types.
"""
construct_solver(::AbstractSolverBlueprint, ::AbstractGMRF) =
    error("construct_solver not implemented for SolverBlueprint")
