# COV_EXCL_START
"""
    GaussianMarkovRandomFieldsEnzyme

Package extension providing Enzyme.jl support for automatic differentiation in
GaussianMarkovRandomFields.jl.

Enzyme differentiates the LLVM IR of whatever it is handed, so anything it can
reach it will attempt — including CHOLMOD's `ccall`s (which carry no derivative
information and yield zeros) and the pure-Julia multifrontal factorization (which
it walks and gets wrong). Every GMRF operation that consumes a factorization
therefore needs an explicit rule here, and any operation *without* one is a
silent-wrong-answer waiting to happen rather than a missing feature.

The rules fall into two groups:

- **Implemented** — `GMRF` construction, `logdetcov` and `logpdf`, each computed
  from a selected inversion instead of by differentiating the factorization.
- **Refusing** — `var`, `gaussian_approximation`, and any GMRF type without the
  accessors in `enzyme/common.jl` raise an `ArgumentError` naming the operation
  and the type, rather than returning a plausible wrong number.

See the "Automatic Differentiation" reference page for the support matrix.

# Example
```julia
using GaussianMarkovRandomFields, Enzyme, LinearSolve

function objective(θ)
    gmrf = GMRF(mean_of(θ), precision_of(θ), LinearSolve.CHOLMODFactorization())
    return logpdf(gmrf, data)
end

grad = autodiff(Reverse, objective, Active, Duplicated(θ, zero(θ)))[1]
```
"""
module GaussianMarkovRandomFieldsEnzyme

using GaussianMarkovRandomFields
const GMRFs = GaussianMarkovRandomFields
using GaussianMarkovRandomFields: GMRF, AbstractGMRF, ChordalGMRF, WorkspaceGMRF,
    MetaGMRF, ConstrainedGMRF, precision_matrix, selinv, gaussian_approximation,
    ObservationLikelihood, loggrad, loghessian, _base_gmrf

using Enzyme
using Distributions: logpdf, logdetcov, mean, var
using SparseArrays
using SparseArrays: getcolptr, AbstractSparseMatrix
using LinearAlgebra
using CliqueTrees.Multifrontal: selinv as mselinv

using Enzyme.EnzymeRules
import Enzyme: Const, Active, Duplicated, Annotation

include("enzyme/common.jl")
include("enzyme/constructors.jl")
include("enzyme/logdetcov.jl")
include("enzyme/logpdf.jl")
include("enzyme/gaussian_approximation.jl")
include("enzyme/unsupported.jl")

end
# COV_EXCL_STOP
