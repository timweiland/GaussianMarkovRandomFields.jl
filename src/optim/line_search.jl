export AbstractLineSearch, BacktrackingLineSearch, NoLineSearch

abstract type AbstractLineSearch end

struct BacktrackingLineSearch <: AbstractLineSearch
    α₀::Real
    τ::Real
    c::Real

    function BacktrackingLineSearch(α₀::Real = 1.0, τ::Real = 0.5, c::Real = 1e-4)
        new(α₀, τ, c)
    end
end

function line_search(
    f::Function,
    ∇f::AbstractVector,
    x::AbstractVector,
    p::AbstractVector,
    ls::BacktrackingLineSearch
)
    α = ls.α₀
    fx = f(x)
    t = -ls.c * dot(∇f, p)
    
    while fx - f(x + α * p) < α * t
        α *= ls.τ
        if α < 1e-8
            break
        end
    end
    return x + α * p
end

struct NoLineSearch <: AbstractLineSearch end

function line_search(
    ::Function,
    ::AbstractVector,
    x::AbstractVector,
    p::AbstractVector,
    ::NoLineSearch
)
    return x + p
end

