using LinearSolve
using LinearAlgebra

export logdet_cov

"""
    logdet_cov(linsolve, alg)

Compute the log determinant of the covariance matrix (inverse of precision).
Dispatches on the algorithm type.
"""
function logdet_cov(linsolve, alg)
    ensure_factorization!(linsolve)
    return _logdet_cov_impl(linsolve, alg)
end

"""
    logdet_cov(linsolve)

Convenience function that dispatches to logdet_cov(linsolve, linsolve.alg).
"""
logdet_cov(linsolve) = logdet_cov(linsolve, linsolve.alg)

# Implementation methods (after factorization is ensured)
_logdet_cov_impl(linsolve, alg) = error("Log determinant of covariance not implemented for algorithm $(typeof(alg))")

function _logdet_cov_impl(linsolve, ::LinearSolve.CHOLMODFactorization)
    factorization = LinearSolve.@get_cacheval(linsolve, :CHOLMODFactorization)
    # Log determinant of covariance = -log determinant of precision
    return -logdet(factorization)
end

function _logdet_cov_impl(linsolve, ::LinearSolve.CholeskyFactorization)
    factorization = LinearSolve.@get_cacheval(linsolve, :CholeskyFactorization)
    # Log determinant of covariance = -log determinant of precision
    return -logdet(factorization)
end

function _logdet_cov_impl(linsolve, ::LinearSolve.DiagonalFactorization)
    # For diagonal matrices, logdet of covariance = -logdet of precision matrix
    # The original matrix is stored in linsolve.A
    return -logdet(linsolve.A)
end

function _logdet_cov_impl(linsolve, ::LinearSolve.PardisoJL)
    # Pardiso logdet - will be implemented in extension
    error("Pardiso logdet implementation requires the Pardiso extension")
end

# Handle DefaultLinearSolver by dispatching on the nested algorithm
function _logdet_cov_impl(linsolve, alg::LinearSolve.DefaultLinearSolver)
    actual_alg = LinearSolve.algchoice_to_alg(Symbol(alg.alg))
    return _logdet_cov_impl(linsolve, actual_alg)
end
