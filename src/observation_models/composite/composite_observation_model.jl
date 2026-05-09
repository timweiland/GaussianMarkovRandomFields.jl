"""
    CompositeObservationModel{T<:Tuple, R<:Tuple} <: ObservationModel

An observation model that combines multiple component observation models.

This type follows the factory pattern - it stores component observation models and
creates `CompositeLikelihood` instances when called with observation data and
hyperparameters.

# Fields
- `components::T`: Tuple of component observation models for type stability
- `routes::R`:     Tuple of per-component kwarg routes. Each entry is either
  `nothing` (forward the full kwargs bag to that component, the default) or a
  `NamedTuple{inner_names}(outer_names)` mapping the component's *internal*
  kwarg names to the outer hyperparameter names exposed by the composite.

# Routing

When two components share an internal kwarg name but should bind to different
outer hyperparameters (e.g. two `ExponentialFamily(Normal)` components needing
distinct `σ` values), supply a route per component:

```julia
m_phys = ExponentialFamily(Normal)
m_data = ExponentialFamily(Normal)
composite = CompositeObservationModel(
    (m_phys, m_data),
    ((σ = :σ_phys,), (σ = :σ_data,)),
)
composite(y; σ_phys = 0.1, σ_data = 5.0)
```

A `nothing` entry preserves the legacy behaviour of forwarding the full kwargs
bag to that component, so the single-arg constructor remains a drop-in for the
common case where component kwarg names are disjoint.

# Example
```julia
gaussian_model = ExponentialFamily(Normal)
poisson_model = ExponentialFamily(Poisson)
composite_model = CompositeObservationModel((gaussian_model, poisson_model))

y_composite = CompositeObservations(([1.0, 2.0], PoissonObservations([3, 4])))
composite_lik = composite_model(y_composite; σ = 1.5)
```
"""
struct CompositeObservationModel{T <: Tuple, R <: Tuple} <: ObservationModel
    components::T
    routes::R

    function CompositeObservationModel(components::T, routes::R) where {T <: Tuple, R <: Tuple}
        if isempty(components)
            throw(ArgumentError("CompositeObservationModel cannot be empty"))
        end
        if length(components) != length(routes)
            throw(
                ArgumentError(
                    "Number of routes ($(length(routes))) must match number of components ($(length(components)))"
                )
            )
        end
        return new{T, R}(components, routes)
    end
end

CompositeObservationModel(components::Tuple) =
    CompositeObservationModel(components, ntuple(_ -> nothing, length(components)))

"""
    CompositeLikelihood{T<:Tuple} <: ObservationLikelihood

A materialized composite likelihood that combines multiple component likelihoods.

Created by calling a `CompositeObservationModel` with observation data and hyperparameters.
Provides efficient evaluation of log-likelihood, gradient, and Hessian by summing
contributions from all component likelihoods.

# Fields
- `components::T`: Tuple of materialized component likelihoods
- `hyperparameter_names::Tuple{Vararg{Symbol}}`: Outer hyperparameter names that were
  used to materialize this likelihood (matches `hyperparameters` on the source model).
"""
struct CompositeLikelihood{T <: Tuple} <: ObservationLikelihood
    components::T
    hyperparameter_names::Tuple{Vararg{Symbol}}

    function CompositeLikelihood(
            components::T,
            hyperparameter_names::Tuple{Vararg{Symbol}} = (),
        ) where {T <: Tuple}
        return new{T}(components, hyperparameter_names)
    end
end

@inline _materialize_component(component, ::Nothing, y_comp, kwargs::NamedTuple) =
    component(y_comp; kwargs...)

@inline function _materialize_component(
        component,
        route::NamedTuple{inner_names, <:Tuple{Vararg{Symbol}}},
        y_comp,
        kwargs::NamedTuple,
    ) where {inner_names}
    inner_vals = map(outer_name -> getfield(kwargs, outer_name), values(route))
    inner = NamedTuple{inner_names}(inner_vals)
    return component(y_comp; inner...)
end

function (composite_model::CompositeObservationModel)(y::CompositeObservations; kwargs...)
    if length(composite_model.components) != length(y.components)
        throw(ArgumentError("Number of model components ($(length(composite_model.components))) must match number of observation components ($(length(y.components)))"))
    end

    nt = values(kwargs)
    component_likelihoods = map(
        composite_model.components, composite_model.routes, y.components,
    ) do model, route, y_comp
        _materialize_component(model, route, y_comp, nt)
    end

    return CompositeLikelihood(component_likelihoods, hyperparameters(composite_model))
end

function hyperparameters(model::CompositeObservationModel)
    out = Symbol[]
    for (component, route) in zip(model.components, model.routes)
        if route === nothing
            for s in hyperparameters(component)
                s in out || push!(out, s)
            end
        else
            for outer in values(route)
                outer in out || push!(out, outer)
            end
        end
    end
    return Tuple(out)
end

hyperparameters(lik::CompositeLikelihood) = lik.hyperparameter_names

# COV_EXCL_START
function Base.show(io::IO, model::CompositeObservationModel)
    n_components = length(model.components)
    print(io, "CompositeObservationModel with $n_components component$(n_components == 1 ? "" : "s"):")
    show_routes = any(r -> r !== nothing, model.routes)
    for (i, component) in enumerate(model.components)
        print(io, "\n  [$i] ")
        show(io, component)
        if show_routes
            route = model.routes[i]
            if route === nothing
                print(io, "  [route: passthrough]")
            else
                pairs_str = join(("$k=:$(getfield(route, k))" for k in keys(route)), ", ")
                print(io, "  [route: ", pairs_str, "]")
            end
        end
    end
    return
end

function Base.show(io::IO, lik::CompositeLikelihood)
    n_components = length(lik.components)
    print(io, "CompositeLikelihood with $n_components component$(n_components == 1 ? "" : "s"):")
    for (i, component) in enumerate(lik.components)
        print(io, "\n  [$i] ")
        show(io, component)
    end
    return
end
# COV_EXCL_STOP
