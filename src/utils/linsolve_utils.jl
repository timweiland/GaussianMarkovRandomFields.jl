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
prepare_for_linsolve(A::SymTridiagonal, alg::LinearSolve.PardisoJL) = prepare_for_linsolve(sparse(A), alg)

# ldlt! doesn't work with `Symmetric`, so keep it as-is
prepare_for_linsolve(A::AbstractMatrix, ::LinearSolve.LDLtFactorization) = A

"""
    configure_algorithm(alg)

Configure algorithm with optimal defaults for GMRF operations.
Default implementation returns the algorithm unchanged.
Specialized methods can modify algorithm parameters for specific solvers.
"""
configure_algorithm(alg) = alg

"""
    algorithm_applicable(alg, A)

Check whether the LinearSolve algorithm `alg` can factorize a matrix of type `typeof(A)`.
Returns `Val{true}()` if applicable, `Val{false}()` otherwise.

GMRF operations routinely change the storage type of the precision matrix: conditioning a
`SymTridiagonal` prior on linear observations yields a sparse posterior precision, for
instance. An algorithm carried over from a prior is only reusable when it is applicable to
the new matrix, so call sites consult this predicate before inheriting one.

Like `supports_selinv` and `supports_backward_solve`, the decision is made at compile time
through dispatch. The default assumes an algorithm is applicable; add a method here for any
solver with genuine restrictions on the matrix type it accepts.
"""
algorithm_applicable(_, _) = Val{true}()

# `LDLtFactorization` bottoms out in `LinearAlgebra.ldlt!`, whose only method taking a bare
# matrix is `ldlt!(::SymTridiagonal)`. Sparse, dense and `Symmetric`-wrapped matrices all
# throw a MethodError, so this algorithm must never be inherited across a change of storage
# type (the CHOLMOD `ldlt!(::Factor, ::AbstractMatrix)` methods do not apply here).
algorithm_applicable(::LinearSolve.LDLtFactorization, ::SymTridiagonal) = Val{true}()
algorithm_applicable(::LinearSolve.LDLtFactorization, ::AbstractMatrix) = Val{false}()

"""
    resolve_linsolve(precision, alg) -> (A, alg_resolved)

Prepare `precision` for LinearSolve and resolve the algorithm that will factorize it.

`alg` is honoured whenever it is applicable to the prepared matrix. Otherwise it is dropped
in favour of `nothing`, which lets LinearSolve auto-select a suitable default, and the
matrix is re-prepared for that default. This keeps fast paths intact — a `SymTridiagonal`
precision with `LDLtFactorization` is passed through untouched — while preventing an
inherited algorithm from being applied to a matrix it cannot handle.
"""
function resolve_linsolve(precision, alg)
    configured_alg = configure_algorithm(alg)
    A = prepare_for_linsolve(precision, configured_alg)
    return _resolve_linsolve(precision, A, configured_alg, algorithm_applicable(configured_alg, A))
end

_resolve_linsolve(_, A, alg, ::Val{true}) = (A, alg)
_resolve_linsolve(precision, _, _, ::Val{false}) = (prepare_for_linsolve(precision, nothing), nothing)
