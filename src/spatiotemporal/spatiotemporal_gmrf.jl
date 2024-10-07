using LinearMaps, LinearAlgebra, SparseArrays

export AbstractSpatiotemporalGMRF,
    time_means, time_vars, time_stds, time_rands, discretization_at_time

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
abstract type AbstractSpatiotemporalGMRF <: AbstractGMRF end

N_t(::AbstractSpatiotemporalGMRF) = error("N_t not implemented")
time_means(::AbstractSpatiotemporalGMRF) = error("time_means not implemented")
time_vars(::AbstractSpatiotemporalGMRF) = error("time_vars not implemented")
time_stds(::AbstractSpatiotemporalGMRF) = error("time_stds not implemented")
time_rands(::AbstractSpatiotemporalGMRF, ::AbstractRNG) =
    error("time_rands not implemented")
discretization_at_time(::AbstractSpatiotemporalGMRF, ::Int) =
    error("discretization_at_time not implemented")
