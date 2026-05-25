export SPDE, ndim, discretize

"""
    SPDE

An abstract type for a stochastic partial differential equation (SPDE).
"""
abstract type SPDE end

function ndim end
function discretize end

ndim(s::SPDE) = throw(MethodError(ndim, (s,))) # COV_EXCL_LINE
# `discretize(::SPDE, ::FEMDiscretization)` fallback lives in `src/fem/types.jl`,
# which is loaded after the `FEMDiscretization` type definition.
