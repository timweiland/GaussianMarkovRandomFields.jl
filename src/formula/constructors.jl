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
    Besag(W)

Formula functor for Besag (intrinsic CAR) random effects.

Usage:
- Create a functor instance: `besag = Besag(W)`
- Use in a formula: `@formula(y ~ 0 + besag(region))`

Note: Calling the functor directly is unsupported outside formula parsing.
"""
struct Besag{WT <: AbstractMatrix}
    W::WT
    normalize_var::Bool
    singleton_policy::Symbol

    function Besag(W::WT; normalize_var::Bool = true, singleton_policy::Symbol = :gaussian) where {WT}
        return new{WT}(W, normalize_var, singleton_policy)
    end
end

(::Besag)(args...) = error("Besag(...) functor is only intended for use inside @formula; not callable directly.")
