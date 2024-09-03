import LinearAlgebra: ldiv!
import Base: \, size

export AbstractPreconditioner

abstract type AbstractPreconditioner end;

ldiv!(y, ::AbstractPreconditioner, x) =
    error("ldiv! not defined for this preconditioner type")
ldiv!(::AbstractPreconditioner, x) = error("ldiv! not defined for this preconditioner type")
\(::AbstractPreconditioner, x) =
    error("left division not defined for this preconditioner type")

Base.size(::AbstractPreconditioner) = error("size not defined for this preconditioner type")
