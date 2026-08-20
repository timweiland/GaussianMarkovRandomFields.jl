module GaussianMarkovRandomFieldsMooncake

using GaussianMarkovRandomFields
using GaussianMarkovRandomFields: hermdiff, ensure_factorization!,
    ensure_loaded!, ensure_numeric!, has_constraints,
    _has_gauss_newton_jacobian, _reverse_mode_gauss_newton_error,
    _constraint_shift, _constraint_log_correction, _constraint_var_correction,
    _logdet_cov_impl, _selinv_diag_impl, _selinv_impl, _backward_solve_impl
using Statistics: mean
using Distributions: logpdf, logdetcov, var
using Mooncake
using Mooncake: @is_primitive, @mooncake_overlay, MinimalCtx, CoDual, NoRData, NoFData, primal, tangent, fdata, zero_tangent
using MooncakeSparse
using SparseArrays: nonzeros, SparseMatrixCSC
using LinearAlgebra: Hermitian, Symmetric, cholesky, diag, dot, logdet, I
using LinearSolve
using LinearSolve: CHOLMODFactorization
using CliqueTrees.Multifrontal: ChordalCholesky
import CliqueTrees.Multifrontal as Multifrontal

# A GMRF whose precision is a plain sparse matrix — the shape produced by the
# CliqueTrees LinearSolve backend. The overlays below additionally require the
# linsolve cache to hold a `ChordalCholesky` (enforced at runtime with an
# actionable error), so other backends fail loudly instead of deep in AD.
const SparseGMRF = GMRF{<:Real, <:AbstractVector, <:Any, <:SparseMatrixCSC}

include("mooncake/chordal.jl")
include("mooncake/gmrf.jl")
include("mooncake/constraints.jl")
include("mooncake/workspace.jl")
include("mooncake/constrained.jl")
include("mooncake/gaussian_approximation.jl")
include("mooncake/unsupported.jl")

end
