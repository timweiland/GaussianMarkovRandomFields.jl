export AbstractSolver, AbstractSolverBlueprint

"""
    AbstractSolver

An abstract type for a solver, which provides methods to compute the mean,
variance, and random samples from a Gaussian Markov Random Field (GMRF).
"""
abstract type AbstractSolver end

"""
    AbstractSolverBlueprint

An abstract type for a blueprint to construct a solver.
A blueprint contains parameters and settings for a solver which are
independent of the concrete GMRF it will be used on.
"""
abstract type AbstractSolverBlueprint end
