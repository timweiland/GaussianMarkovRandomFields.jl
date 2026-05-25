using LinearMaps, LinearAlgebra, SparseArrays

export AbstractSpatiotemporalGMRF,
    time_means, time_vars, time_stds, time_rands, discretization_at_time, N_t

################################################################################
#
#    AbstractSpatiotemporalGMRF
#
#    Abstract type for spatiotemporal Gaussian Markov Random Fields.
#    Each subtype must implement the following methods:
#    - N_t
#    - time_means
#    - time_vars
#    - time_stds
#    - time_rands
#    - discretization_at_time
#
################################################################################
"""
    AbstractSpatiotemporalGMRF

A spatiotemporal GMRF is a GMRF that explicitly encodes the spatial and temporal
structure of the underlying random field.
All time points are modelled in one joint GMRF.
It provides utilities to get statistics, draw samples and get the spatial discretization
at a given time.
"""
abstract type AbstractSpatiotemporalGMRF{T <: Real, PrecisionMap <: LinearMap{T}} <: AbstractGMRF{T, PrecisionMap} end

"""
    N_t(::AbstractSpatiotemporalGMRF)

Return the number of time points in the spatiotemporal GMRF.
"""
N_t(d::AbstractSpatiotemporalGMRF) = throw(MethodError(N_t, (d,))) # COV_EXCL_LINE

"""
    time_means(::AbstractSpatiotemporalGMRF)

Return the means of the spatiotemporal GMRF at each time point.

# Returns
- A vector of means of length Nₜ, one for each time point.
"""
time_means(d::AbstractSpatiotemporalGMRF) = throw(MethodError(time_means, (d,))) # COV_EXCL_LINE

"""
    time_vars(::AbstractSpatiotemporalGMRF)

Return the marginal variances of the spatiotemporal GMRF at each time point.

# Returns
- A vector of marginal variances of length Nₜ, one for each time point.
"""
time_vars(d::AbstractSpatiotemporalGMRF) = throw(MethodError(time_vars, (d,))) # COV_EXCL_LINE

"""
    time_stds(::AbstractSpatiotemporalGMRF)

Return the marginal standard deviations of the spatiotemporal GMRF at each time point.

# Returns
- A vector of marginal standard deviations of length Nₜ, one for each time point.
"""
time_stds(d::AbstractSpatiotemporalGMRF) = throw(MethodError(time_stds, (d,))) # COV_EXCL_LINE

"""
    time_rands(::AbstractSpatiotemporalGMRF, rng::AbstractRNG)

Draw samples from the spatiotemporal GMRF at each time point.

# Returns
- A vector of sample values of length Nₜ, one sample value vector
  for each time point.
"""
# COV_EXCL_START
time_rands(d::AbstractSpatiotemporalGMRF, rng::AbstractRNG) =
    throw(MethodError(time_rands, (d, rng)))
# COV_EXCL_STOP

"""
    discretization_at_time(::AbstractSpatiotemporalGMRF, t::Int)

Return the spatial discretization at time `t`.
"""
# COV_EXCL_START
discretization_at_time(d::AbstractSpatiotemporalGMRF, t::Int) =
    throw(MethodError(discretization_at_time, (d, t)))
# COV_EXCL_STOP
