export LinearMapWithSqrt
import LinearMaps: _unsafe_mul!

struct LinearMapWithSqrt{T} <: LinearMaps.LinearMap{T}
    A::LinearMaps.LinearMap{T}
    A_sqrt::LinearMaps.LinearMap{T}

    function LinearMapWithSqrt(
        A::LinearMaps.LinearMap{T},
        A_sqrt::LinearMaps.LinearMap{T},
    ) where {T}
        Base.size(A, 1) == Base.size(A_sqrt, 1) || throw(ArgumentError("size mismatch"))
        new{T}(A, A_sqrt)
    end
end

function LinearMaps._unsafe_mul!(y, L::LinearMapWithSqrt, x::AbstractVector)
    LinearMaps._unsafe_mul!(y, L.A, x)
end

function LinearMaps.size(L::LinearMapWithSqrt)
    return LinearMaps.size(L.A)
end

function LinearMaps.adjoint(L::LinearMapWithSqrt)
    return L
end

function LinearMaps.transpose(L::LinearMapWithSqrt)
    return L
end

function to_matrix(L::LinearMapWithSqrt)
    return to_matrix(L.A)
end

function linmap_sqrt(L::LinearMapWithSqrt)
    return L.A_sqrt
end
