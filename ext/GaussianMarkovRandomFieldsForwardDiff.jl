"""
    GaussianMarkovRandomFieldsForwardDiff

"""
module GaussianMarkovRandomFieldsForwardDiff

import GaussianMarkovRandomFields as GMRFs
import GaussianMarkovRandomFields: GMRF, ADJacobianMap
import Distributions: logdetcov

using ForwardDiff
using LinearAlgebra
using LinearMaps
import LinearMaps: _unsafe_mul!
using LinearSolve
using SparseArrays

function LinearMaps._unsafe_mul!(y, J::ADJacobianMap, x::AbstractVector)
    g(t) = J.f(J.x₀ + t * x)
    return y .= ForwardDiff.derivative(g, 0.0)
end

include("forwarddiff/common.jl")
include("forwarddiff/gmrf_constructors.jl")
include("forwarddiff/gmrf_gaussian_approximation.jl")
include("forwarddiff/constrained_gmrf.jl")
include("forwarddiff/workspace_constructors.jl")
include("forwarddiff/constrained_workspace.jl")
include("forwarddiff/workspace_gaussian_approximation.jl")
include("forwarddiff/logdetcov.jl")
include("forwarddiff/autodiff_likelihood_dual.jl")
include("forwarddiff/autodiff_likelihood_ift.jl")

end
