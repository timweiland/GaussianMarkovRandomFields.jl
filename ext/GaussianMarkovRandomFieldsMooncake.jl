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

# --- On the `COV_EXCL_START`/`COV_EXCL_STOP` regions in the files below ---
#
# Mooncake does not call an `@mooncake_overlay` body the way Julia calls a
# function. It derives a rule from the body's IR and runs that instead, so the
# original statements never execute and the line counters Julia's coverage
# instrumentation puts in them never fire. Anything reachable only through an
# overlay is invisible to coverage for the same reason.
#
# Verified rather than assumed: an explicit `Ref` increment inserted into
# `_mooncake_constraint_schur` fires on a direct primal call and does NOT fire
# during a Mooncake gradient, while that gradient still matches FiniteDiff to
# 1e-4 — so the code is exercised, it is only unobservable. No amount of extra
# testing moves those lines, which is why the regions are excluded rather than
# covered.
#
# The exclusions are deliberately narrow. `Mooncake.rrule!!` bodies ARE called
# as ordinary functions and stay measured — chordal.jl and unsupported.jl are
# both at 100%, and coverage remains a real gate for them. What the excluded
# regions give up is coverage as a signal for the overlay paths; correctness
# there is gated instead by the finite-difference comparisons in
# test/autodiff/test_cliquetrees_mooncake.jl, which fail loudly if an overlay
# stops being reached.

include("mooncake/unsupported.jl")
include("mooncake/chordal.jl")
include("mooncake/gmrf.jl")
include("mooncake/constraints.jl")
include("mooncake/workspace.jl")
include("mooncake/constrained.jl")
include("mooncake/gaussian_approximation.jl")

end
