using LinearSolve, SparseArrays

"""
    prepare_for_linsolve(A, alg)

Prepare matrix A for LinearSolve based on the algorithm type.
Default implementation uses symmetrize() for proper symmetric handling.
Specialized methods for algorithms that need raw matrices can override this.
"""
prepare_for_linsolve(A::AbstractMatrix, _) = symmetrize(A)

prepare_for_linsolve(A::LinearMaps.LinearMap, alg) = prepare_for_linsolve(to_matrix(A), alg)

# Pardiso algorithms need raw sparse matrices, not Symmetric wrappers
prepare_for_linsolve(A::AbstractMatrix, ::LinearSolve.PardisoJL) = tril(sparse(A))

# SymTridiagonal only works nicely with LDLt
prepare_for_linsolve(A::SymTridiagonal, ::LinearSolve.LDLtFactorization) = A
prepare_for_linsolve(A::SymTridiagonal, alg) = prepare_for_linsolve(sparse(A), alg)
prepare_for_linsolve(A::SymTridiagonal, ::LinearSolve.PardisoJL) = prepare_for_linsolve(sparse(A), alg)

# ldlt! doesn't work with `Symmetric`, so keep it as-is
prepare_for_linsolve(A::AbstractMatrix, ::LinearSolve.LDLtFactorization) = A

"""
    configure_algorithm(alg)

Configure algorithm with optimal defaults for GMRF operations.
Default implementation returns the algorithm unchanged.
Specialized methods can modify algorithm parameters for specific solvers.
"""
configure_algorithm(alg) = alg
