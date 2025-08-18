using LinearAlgebra

"""
    symmetrize(A)

Apply appropriate symmetric wrapper to matrix types.
- `Diagonal`: Return as-is (no wrapping needed)
- `Tridiagonal`: Convert to `SymTridiagonal`
- Others: Wrap in `Symmetric`
"""
symmetrize(A::Diagonal) = A
symmetrize(A::SymTridiagonal) = A
symmetrize(A::Tridiagonal) = SymTridiagonal(A)
symmetrize(A::AbstractMatrix) = Symmetric(A)
