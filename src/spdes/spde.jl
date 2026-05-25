export SPDE, ndim, discretize

"""
    SPDE
    
An abstract type for a stochastic partial differential equation (SPDE).
"""
abstract type SPDE end

ndim(s::SPDE) = throw(MethodError(ndim, (s,))) # COV_EXCL_LINE
discretize(s::SPDE, d::FEMDiscretization) = throw(MethodError(discretize, (s, d))) # COV_EXCL_LINE
