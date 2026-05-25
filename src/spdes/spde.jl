export SPDE, ndim, discretize

"""
    SPDE

An abstract type for a stochastic partial differential equation (SPDE).
"""
abstract type SPDE end

function ndim end
function discretize end

ndim(::SPDE) = error("dim not implemented for SPDE")  # COV_EXCL_LINE
