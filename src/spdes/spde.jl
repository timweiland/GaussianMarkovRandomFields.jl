export SPDE, ndim, intrinsic_dim, discretize

"""
    SPDE

An abstract type for a stochastic partial differential equation (SPDE).
"""
abstract type SPDE end

function ndim end
function discretize end
