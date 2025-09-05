export IID, RandomWalk, AR1, Besag, build_formula_components

"""
    build_formula_components(formula, data; kwargs...)

Placeholder function for the formula interface. A concrete method is provided
by the `GaussianMarkovRandomFieldsFormula` extension when `StatsModels` is loaded.
"""
function build_formula_components end

"""
    IID(group)

Formula constructor for IID random effects. Use only inside `@formula`.
"""
function IID(args...)
    error("IID(...) is only intended for use inside @formula; load StatsModels to activate the formula interface.")
end

"""
    RandomWalk(order, index)

Formula constructor for RandomWalk random effects (order=1 MVP). Use only inside `@formula`.
"""
function RandomWalk(args...)
    error("RandomWalk(...) is only intended for use inside @formula; load StatsModels to activate the formula interface.")
end

"""
    AR1(index)

Formula constructor for AR1 random effects. Use only inside `@formula`.
"""
function AR1(args...)
    error("AR1(...) is only intended for use inside @formula; load StatsModels to activate the formula interface.")
end

"""
    Besag(W; id_to_node = nothing, normalize_var = true, singleton_policy = :gaussian)

Formula functor for Besag (intrinsic CAR) random effects.

Usage:
- Create a functor instance: `besag = Besag(W)`
- With string/categorical region IDs: `besag = Besag(W; id_to_node = Dict("WesternIsles" => 11, ...))`
- Use in a formula: `@formula(y ~ 0 + besag(region))`

Notes
- `id_to_node` maps arbitrary region identifiers to integer node indices (1-based) of `W`.
- Calling the functor directly is unsupported outside formula parsing.
"""
struct Besag{WT <: AbstractMatrix, MT}
    W::WT
    id_to_node::MT  # may be Nothing or a mapping supporting getindex
    normalize_var::Bool
    singleton_policy::Symbol

    function Besag(
            W::WT; id_to_node = nothing, normalize_var::Bool = true, singleton_policy::Symbol = :gaussian
        ) where {WT}
        return new{WT, typeof(id_to_node)}(W, id_to_node, normalize_var, singleton_policy)
    end
end

(::Besag)(args...) = error("Besag(...) functor is only intended for use inside @formula; not callable directly.")
