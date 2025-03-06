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

ldiv!(y, ::AbstractPreconditioner, x::AbstractVector) =
    error("ldiv! not defined for this preconditioner type")
ldiv!(::AbstractPreconditioner, x::AbstractVector) =
    error("ldiv! not defined for this preconditioner type")
\(::AbstractPreconditioner, x::AbstractVector) =
    error("left division not defined for this preconditioner type")

Base.size(::AbstractPreconditioner) = error("size not defined for this preconditioner type")
