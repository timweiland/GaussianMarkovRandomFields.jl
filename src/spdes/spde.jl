export SPDE, ndim, discretize

"""
    SPDE
    
An abstract type for a stochastic partial differential equation (SPDE).
"""
abstract type SPDE end

ndim(s::SPDE) = throw(MethodError(ndim, (s,)))
discretize(s::SPDE, d::FEMDiscretization) = throw(MethodError(discretize, (s, d)))
