import LinearAlgebra: ldiv!
import Base: \, size

export AbstractPreconditioner

abstract type AbstractPreconditioner end;

ldiv!(y, ::AbstractPreconditioner, x::AbstractVector) =
    error("ldiv! not defined for this preconditioner type")
ldiv!(::AbstractPreconditioner, x::AbstractVector) =
    error("ldiv! not defined for this preconditioner type")
\(::AbstractPreconditioner, x::AbstractVector) =
    error("left division not defined for this preconditioner type")

Base.size(::AbstractPreconditioner) = error("size not defined for this preconditioner type")
