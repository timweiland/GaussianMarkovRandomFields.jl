export AbstractVarianceStrategy

# Legacy solver types removed - now using LinearSolve.jl directly

"""
    AbstractVarianceStrategy

An abstract type for a strategy to compute the variance of a GMRF.
"""
abstract type AbstractVarianceStrategy end
