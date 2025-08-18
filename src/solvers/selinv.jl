using LinearSolve
using LinearAlgebra
using SelectedInversion

export selinv

"""
    supports_selinv(alg)

Check if a LinearSolve algorithm supports selected inversion (selinv) operations.
Returns `Val{true}()` if supported, `Val{false}()` otherwise.

This is determined at compile-time through dispatch on the algorithm type.
"""
supports_selinv(::LinearSolve.CHOLMODFactorization) = Val{true}()
supports_selinv(::LinearSolve.CholeskyFactorization) = Val{true}()
supports_selinv(::LinearSolve.DiagonalFactorization) = Val{true}()
supports_selinv(::LinearSolve.PardisoJL) = Val{true}()

# Handle DefaultLinearSolver by converting to actual algorithm type
function supports_selinv(alg::LinearSolve.DefaultLinearSolver)
    actual_alg = LinearSolve.algchoice_to_alg(Symbol(alg.alg))
    return supports_selinv(actual_alg)
end

# Fallback for all other algorithms
supports_selinv(::Any) = Val{false}()

"""
    selinv_diag(linsolve, alg)

Compute the diagonal of the inverse matrix using selected inversion.
Dispatches on the algorithm type.
"""
function selinv_diag(linsolve, alg)
    ensure_factorization!(linsolve)
    return _selinv_diag_impl(linsolve, alg)
end

"""
    selinv_diag(linsolve)

Convenience function that dispatches to selinv_diag(linsolve, linsolve.alg).
"""
selinv_diag(linsolve) = selinv_diag(linsolve, linsolve.alg)

"""
    selinv(linsolve, alg)

Compute the full selected inverse matrix using selected inversion.
Dispatches on the algorithm type.
"""
function selinv(linsolve::LinearSolve.LinearCache, alg)
    ensure_factorization!(linsolve)
    return _selinv_impl(linsolve, alg)
end

"""
    selinv(linsolve)

Convenience function that dispatches to selinv(linsolve, linsolve.alg).
"""
selinv(linsolve::LinearSolve.LinearCache) = selinv(linsolve, linsolve.alg)

# Implementation methods (after factorization is ensured)
_selinv_diag_impl(linsolve, alg) = error("Selected inversion not implemented for algorithm $(typeof(alg))")

function _selinv_diag_impl(linsolve, ::LinearSolve.CHOLMODFactorization)
    factorization = LinearSolve.@get_cacheval(linsolve, :CHOLMODFactorization)
    return SelectedInversion.selinv_diag(factorization)
end

function _selinv_diag_impl(linsolve, ::LinearSolve.CholeskyFactorization)
    factorization = LinearSolve.@get_cacheval(linsolve, :CholeskyFactorization)
    return SelectedInversion.selinv_diag(factorization)
end

function _selinv_diag_impl(linsolve, ::LinearSolve.DiagonalFactorization)
    # For diagonal matrices, selinv is just pointwise inversion of diagonal elements
    # The original matrix is stored in linsolve.A
    return 1 ./ diag(linsolve.A)
end

function _selinv_diag_impl(linsolve, ::LinearSolve.PardisoJL)
    # Pardiso selected inversion - will be implemented in extension
    error("Pardiso selinv implementation requires the Pardiso extension")
end

# Handle DefaultLinearSolver by dispatching on the nested algorithm
function _selinv_diag_impl(linsolve, alg::LinearSolve.DefaultLinearSolver)
    actual_alg = LinearSolve.algchoice_to_alg(Symbol(alg.alg))
    return _selinv_diag_impl(linsolve, actual_alg)
end

# Implementation methods for full selected inverse
_selinv_impl(linsolve, alg) = error("Full selected inversion not implemented for algorithm $(typeof(alg))")

function _selinv_impl(linsolve, ::LinearSolve.CHOLMODFactorization)
    factorization = LinearSolve.@get_cacheval(linsolve, :CHOLMODFactorization)
    return SelectedInversion.selinv(factorization; depermute = true).Z
end

function _selinv_impl(linsolve, ::LinearSolve.CholeskyFactorization)
    factorization = LinearSolve.@get_cacheval(linsolve, :CholeskyFactorization)
    return SelectedInversion.selinv(factorization; depermute = true).Z
end

function _selinv_impl(linsolve, ::LinearSolve.DiagonalFactorization)
    # For diagonal matrices, full selected inverse is just the diagonal inverse
    return spdiagm(0 => 1 ./ diag(linsolve.A))
end

function _selinv_impl(linsolve, ::LinearSolve.PardisoJL)
    # Pardiso selected inversion - will be implemented in extension
    error("Pardiso full selinv implementation requires the Pardiso extension")
end

# Handle DefaultLinearSolver by dispatching on the nested algorithm
function _selinv_impl(linsolve, alg::LinearSolve.DefaultLinearSolver)
    actual_alg = LinearSolve.algchoice_to_alg(Symbol(alg.alg))
    return _selinv_impl(linsolve, actual_alg)
end
