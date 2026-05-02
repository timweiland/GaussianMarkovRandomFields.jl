"""
    GaussianMarkovRandomFieldsForwardDiff

"""
module GaussianMarkovRandomFieldsForwardDiff

import GaussianMarkovRandomFields as GMRFs
import GaussianMarkovRandomFields: GMRF
import Distributions: logdetcov

using ForwardDiff
using LinearAlgebra
using LinearMaps
using LinearSolve
using SparseArrays

include("forwarddiff/common.jl")
include("forwarddiff/gmrf_constructors.jl")
include("forwarddiff/gmrf_gaussian_approximation.jl")
include("forwarddiff/constrained_gmrf.jl")
include("forwarddiff/workspace_constructors.jl")
include("forwarddiff/constrained_workspace.jl")
include("forwarddiff/workspace_gaussian_approximation.jl")
include("forwarddiff/logdetcov.jl")
include("forwarddiff/pointwise_hessian.jl")
include("forwarddiff/autodiff_likelihood_dual.jl")

end
