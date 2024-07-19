export SPDE, ndim, discretize

"""
    SPDE
    
An abstract type for a stochastic partial differential equation (SPDE).
"""
abstract type SPDE end

ndim(::SPDE) = error("dim not implemented for SPDE")
discretize(::SPDE, ::FEMDiscretization) = error("discretize not implemented for SPDE")
