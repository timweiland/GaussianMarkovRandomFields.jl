import LinearAlgebra: ldiv!
import Base: \, size

export AbstractPreconditioner

@doc raw"""
    AbstractPreconditioner

Abstract type for preconditioners.
Should implement the following methods:
- ldiv!(y, P::AbstractPreconditioner, x::AbstractVector)
- ldiv!(P::AbstractPreconditioner, x::AbstractVector)
- \\(P::AbstractPreconditioner, x::AbstractVector)
- size(P::AbstractPreconditioner)
"""
abstract type AbstractPreconditioner{T} end;

ldiv!(y, P::AbstractPreconditioner, x::AbstractVector) =
    throw(MethodError(ldiv!, (y, P, x)))
ldiv!(P::AbstractPreconditioner, x::AbstractVector) =
    throw(MethodError(ldiv!, (P, x)))
\(P::AbstractPreconditioner, x::AbstractVector) =
    throw(MethodError(\, (P, x)))

Base.size(P::AbstractPreconditioner) = throw(MethodError(size, (P,)))
