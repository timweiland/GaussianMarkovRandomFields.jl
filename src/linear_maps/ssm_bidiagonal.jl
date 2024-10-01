using LinearMaps

export SSMBidiagonalMap

struct SSMBidiagonalMap{T} <: LinearMaps.LinearMap{T}
    A::LinearMaps.LinearMap{T}
    B::LinearMaps.LinearMap{T}
    C::LinearMaps.LinearMap{T}
    N::Int
    M::Int
    N_t::Int
    main_diag::LinearMaps.LinearMap{T}
    off_diag::LinearMaps.LinearMap{T}

    function SSMBidiagonalMap(
        A::LinearMaps.LinearMap{T},
        B::LinearMaps.LinearMap{T},
        C::LinearMaps.LinearMap{T},
        N_t::Int,
    ) where {T}
        (N_t > 1) || throw(ArgumentError("N_t must be greater than 1"))
        size(B, 2) == size(A, 2) || throw(ArgumentError("size mismatch"))
        size(C) == size(B) || throw(ArgumentError("size mismatch"))
        N = size(A, 1) + (N_t - 1) * size(B, 1)
        M = N_t * size(B, 2)
        main_diag = blockdiag(A, repeat([C], N_t - 2)..., ZeroMap{Float64}(size(C)...))
        off_diag = blockdiag(repeat([B], N_t - 1)...)
        new{T}(A, B, C, N, M, N_t, main_diag, off_diag)
    end
end

function LinearMaps._unsafe_mul!(y, L::SSMBidiagonalMap, x::AbstractVector)
    y .= 0
    y += L.main_diag * x
    y[(Base.size(L.A, 1)+1):end] .+= L.off_diag * x[1:(end-Base.size(L.B, 2))]
    return y
end

function LinearMaps.size(L::SSMBidiagonalMap)
    return (L.N, L.M)
end

function LinearMaps._unsafe_mul!(
    y,
    L::LinearMaps.TransposeMap{<:Any,<:SSMBidiagonalMap},
    x::AbstractVector,
)
    y .= 0
    y += L.lmap.main_diag' * x
    y[1:(end-Base.size(L.lmap.B, 2))] .+=
        L.lmap.off_diag' * x[(Base.size(L.lmap.A, 1)+1):end]
    return y
end
