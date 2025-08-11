using LinearSolve
using LinearAlgebra
using SelectedInversion

export supports_selinv, supports_backward_solve, selinv_diag, backward_solve, ensure_factorization!

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
    supports_backward_solve(alg)

Check if a LinearSolve algorithm supports backward solve operations.
Returns `Val{true}()` if supported, `Val{false}()` otherwise.

This is determined at compile-time through dispatch on the algorithm type.
"""
supports_backward_solve(::LinearSolve.CHOLMODFactorization) = Val{true}()
supports_backward_solve(::LinearSolve.CholeskyFactorization) = Val{true}()
supports_backward_solve(::LinearSolve.DiagonalFactorization) = Val{true}()
supports_backward_solve(::LinearSolve.PardisoJL) = Val{true}()

# Handle DefaultLinearSolver by converting to actual algorithm type
function supports_backward_solve(alg::LinearSolve.DefaultLinearSolver)
    actual_alg = LinearSolve.algchoice_to_alg(Symbol(alg.alg))
    return supports_backward_solve(actual_alg)
end

# Fallback for all other algorithms
supports_backward_solve(::Any) = Val{false}()

"""
    ensure_factorization!(linsolve)

Ensure that the LinearSolve cache has computed its factorization.
This is needed before accessing cached factorization data.
"""
function ensure_factorization!(linsolve)
    if linsolve.isfresh
        LinearSolve.solve!(linsolve)
    end
    return nothing
end

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

"""
    backward_solve(linsolve, x, alg)

Perform backward solve L^T \\ x where L is the Cholesky factor.
Dispatches on the algorithm type.
"""
function backward_solve(linsolve, x, alg)
    ensure_factorization!(linsolve)
    return _backward_solve_impl(linsolve, x, alg)
end

"""
    backward_solve(linsolve, x)

Convenience function that dispatches to backward_solve(linsolve, x, linsolve.alg).
"""
backward_solve(linsolve, x) = backward_solve(linsolve, x, linsolve.alg)

# Implementation methods (after factorization is ensured)
_backward_solve_impl(linsolve, x, alg) = error("Backward solve not implemented for algorithm $(typeof(alg))")

function _backward_solve_impl(linsolve, x, ::LinearSolve.CHOLMODFactorization)
    factorization = LinearSolve.@get_cacheval(linsolve, :CHOLMODFactorization)
    return factorization.UP \ x
end

function _backward_solve_impl(linsolve, x, ::LinearSolve.CholeskyFactorization)
    factorization = LinearSolve.@get_cacheval(linsolve, :CholeskyFactorization)
    return factorization.U \ x
end

function _backward_solve_impl(linsolve, x, ::LinearSolve.DiagonalFactorization)
    # For diagonal matrices, "backward solve" with inverse square root
    # We need Q^(-1/2) * x = (1/sqrt(diag)) .* x
    return x ./ sqrt.(diag(linsolve.A))
end

function _backward_solve_impl(linsolve, x, ::LinearSolve.PardisoJL)
    # Pardiso backward solve - will be implemented in extension
    error("Pardiso backward solve implementation requires the Pardiso extension")
end

# Handle DefaultLinearSolver by dispatching on the nested algorithm
function _backward_solve_impl(linsolve, x, alg::LinearSolve.DefaultLinearSolver)
    actual_alg = LinearSolve.algchoice_to_alg(Symbol(alg.alg))
    return _backward_solve_impl(linsolve, x, actual_alg)
end
