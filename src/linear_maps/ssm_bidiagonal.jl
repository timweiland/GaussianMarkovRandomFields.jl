using LinearMaps

export SSMBidiagonalMap

# TODO: Better naming for this?
@doc raw"""
    SSMBidiagonalMap{T}(
        A::LinearMap{T},
        B::LinearMap{T},
        C::LinearMap{T},
        N_t::Int,
    )

Represents the block-bidiagonal map given by the (N_t) x (N_t - 1) sized
block structure:

```math
\begin{bmatrix}
A & 0 & \cdots & 0 \\
B & C & \cdots & 0 \\
0 & B & C & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & B
\end{bmatrix}
```

which occurs as a square root in the discretization of GMRF-based state-space
models. `N_t` is the total number of blocks along the rows.
"""
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
        M = (N_t - 1) * size(B, 2)
        main_diag = blockdiag(A, repeat([C], N_t - 2)...)
        off_diag = blockdiag(repeat([B], N_t - 1)...)
        return new{T}(A, B, C, N, M, N_t, main_diag, off_diag)
    end
end

function LinearMaps._unsafe_mul!(y, L::SSMBidiagonalMap, x::AbstractVector)
    y .= 0
    y[1:(end - Base.size(L.B, 1))] += L.main_diag * x
    y[(Base.size(L.A, 1) + 1):end] .+= L.off_diag * x
    return y
end

function LinearMaps.size(L::SSMBidiagonalMap)
    return (L.N, L.M)
end

function LinearMaps._unsafe_mul!(
        y,
        L::LinearMaps.TransposeMap{<:Any, <:SSMBidiagonalMap},
        x::AbstractVector,
    )
    y .= 0
    y += L.lmap.main_diag' * x[1:(end - Base.size(L.lmap.B, 1))]
    y .+= L.lmap.off_diag' * x[(Base.size(L.lmap.A, 1) + 1):end]
    return y
end
