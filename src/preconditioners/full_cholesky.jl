using LinearAlgebra, SparseArrays
import LinearAlgebra: ldiv!
import Base: \, size

export FullCholeskyPreconditioner

"""
    FullCholeskyPreconditioner

A preconditioner that uses a full Cholesky factorization of the matrix,
i.e. P = A, so P⁻¹ = A⁻¹.
Does not make sense to use on its own, but can be used as a building block
for more complex preconditioners.
"""
struct FullCholeskyPreconditioner <: AbstractPreconditioner
    cho::Union{Cholesky,SparseArrays.CHOLMOD.Factor}

    function FullCholeskyPreconditioner(A::AbstractMatrix)
        cho = cholesky(Symmetric(A))
        new(cho)
    end

    function FullCholeskyPreconditioner(cho::Union{Cholesky,SparseArrays.CHOLMOD.Factor})
        new(cho)
    end
end

ldiv!(y, P::FullCholeskyPreconditioner, x::AbstractVector) = (y .= P.cho \ x)
ldiv!(P::FullCholeskyPreconditioner, x::AbstractVector) = (x .= P.cho \ x)
\(P::FullCholeskyPreconditioner, x::AbstractVector) = P.cho \ x
Base.size(P::FullCholeskyPreconditioner) = size(P.cho)
Base.size(P::FullCholeskyPreconditioner, i) = size(P.cho, i)
