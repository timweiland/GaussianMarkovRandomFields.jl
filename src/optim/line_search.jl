export AbstractLineSearch, BacktrackingLineSearch, NoLineSearch

"""
    AbstractLineSearch

Abstract type for the specification of a line search scheme for optimization.
"""
abstract type AbstractLineSearch end

"""
    BacktrackingLineSearch

Specification of a line search based on backtracking via the Armijo condition.
TODO: Description of parameters
"""
struct BacktrackingLineSearch <: AbstractLineSearch
    α₀::Real
    τ::Real
    c::Real

    function BacktrackingLineSearch(α₀::Real = 1.0, τ::Real = 0.5, c::Real = 1.0e-4)
        return new(α₀, τ, c)
    end
end

function line_search(
        f::Function,
        ∇f::AbstractVector,
        x::AbstractVector,
        p::AbstractVector,
        ls::BacktrackingLineSearch,
    )
    α = ls.α₀
    fx = f(x)
    t = -ls.c * dot(∇f, p)

    while fx - f(x + α * p) < α * t
        α *= ls.τ
        if α < 1.0e-8
            break
        end
    end
    return x + α * p
end

"""
    NoLineSearch

A type that communicates that no line search will be used, i.e. the initial step
proposed by the optimization algorithm is the step that will be taken.
"""
struct NoLineSearch <: AbstractLineSearch end

function line_search(
        ::Function,
        ::AbstractVector,
        x::AbstractVector,
        p::AbstractVector,
        ::NoLineSearch,
    )
    return x + p
end
