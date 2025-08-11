using LinearSolve
using LinearAlgebra

export supports_backward_solve, backward_solve

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