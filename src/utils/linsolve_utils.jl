using LinearSolve, SparseArrays

"""
    prepare_for_linsolve(A, alg)

Prepare matrix A for LinearSolve based on the algorithm type.
Default implementation uses symmetrize() for proper symmetric handling.
Specialized methods for algorithms that need raw matrices can override this.
"""
prepare_for_linsolve(A::AbstractMatrix, alg) = symmetrize(A)

# Pardiso algorithms need raw sparse matrices, not Symmetric wrappers
prepare_for_linsolve(A::AbstractMatrix, ::LinearSolve.PardisoJL) = tril(sparse(A))

"""
    configure_algorithm(alg)

Configure algorithm with optimal defaults for GMRF operations.
Default implementation returns the algorithm unchanged.
Specialized methods can modify algorithm parameters for specific solvers.
"""
configure_algorithm(alg) = alg
