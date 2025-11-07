export IID, RandomWalk, AR1, Besag, BYM2, build_formula_components

"""
    build_formula_components(formula, data; kwargs...)

Placeholder function for the formula interface. A concrete method is provided
by the `GaussianMarkovRandomFieldsFormula` extension when `StatsModels` is loaded.
"""
function build_formula_components end

"""
    IID(; constraint = nothing)

Formula functor for IID (independent and identically distributed) random effects.

# Arguments
- `constraint`: Optional constraint specification. Can be:
  - `nothing` (default): No constraints
  - `:sumtozero`: Sum-to-zero constraint for identifiability
  - `(A, e)`: Custom linear constraint matrix and vector where `Ax = e`

# Usage
```julia
# Unconstrained IID effect
iid = IID()
@formula(y ~ 0 + iid(group))

# With sum-to-zero constraint
iid_sz = IID(constraint=:sumtozero)
@formula(y ~ 0 + iid_sz(group))

# With custom constraint
A = [1.0 1.0 0.0]  # x1 + x2 = 0
e = [0.0]
iid_custom = IID(constraint=(A, e))
@formula(y ~ 0 + iid_custom(group))
```

# Notes
- You must create an IID instance before using it in a formula
- Calling the functor directly is unsupported outside formula parsing
"""
struct IID
    constraint::Union{Nothing, Symbol, Tuple{AbstractMatrix, AbstractVector}}

    function IID(; constraint = nothing)
        return new(constraint)
    end
end

(::IID)(args...) = error("IID(...) functor is only intended for use inside @formula; not callable directly.")

"""
    RandomWalk(; additional_constraints = nothing)

Formula functor for RandomWalk random effects (order=1 supported).

# Arguments
- `additional_constraints`: Optional additional constraints beyond the built-in sum-to-zero constraint.
  Can be:
  - `nothing` (default): Only the built-in sum-to-zero constraint
  - `(A, e)`: Custom additional linear constraint matrix and vector where `Ax = e`

# Usage
```julia
# RW1 with built-in sum-to-zero constraint
rw1 = RandomWalk()
@formula(y ~ 0 + rw1(1, time))

# RW1 with additional constraints
A = [1.0 0.0 1.0 zeros(7)...]  # x1 + x3 = 0 (in addition to sum-to-zero)
e = [0.0]
rw1_extra = RandomWalk(additional_constraints=(A, e))
@formula(y ~ 0 + rw1_extra(1, time))
```

# Notes
- RW1 always has a built-in sum-to-zero constraint for identifiability
- Use `additional_constraints` to specify constraints beyond sum-to-zero
- You must create a RandomWalk instance before using it in a formula
- Calling the functor directly is unsupported outside formula parsing
"""
struct RandomWalk
    additional_constraints::Union{Nothing, Tuple{AbstractMatrix, AbstractVector}}

    function RandomWalk(; additional_constraints = nothing)
        return new(additional_constraints)
    end
end

(::RandomWalk)(args...) = error("RandomWalk(...) functor is only intended for use inside @formula; not callable directly.")

"""
    AR1(; constraint = nothing)

Formula functor for AR1 (first-order autoregressive) random effects.

# Arguments
- `constraint`: Optional constraint specification. Can be:
  - `nothing` (default): No constraints
  - `:sumtozero`: Sum-to-zero constraint for identifiability
  - `(A, e)`: Custom linear constraint matrix and vector where `Ax = e`

# Usage
```julia
# Unconstrained AR1 effect
ar1 = AR1()
@formula(y ~ 0 + ar1(time))

# With sum-to-zero constraint
ar1_sz = AR1(constraint=:sumtozero)
@formula(y ~ 0 + ar1_sz(time))

# With custom constraint
A = [1.0 1.0 zeros(8)...]  # First two time points sum to zero
e = [0.0]
ar1_custom = AR1(constraint=(A, e))
@formula(y ~ 0 + ar1_custom(time))
```

# Notes
- You must create an AR1 instance before using it in a formula
- Calling the functor directly is unsupported outside formula parsing
"""
struct AR1
    constraint::Union{Nothing, Symbol, Tuple{AbstractMatrix, AbstractVector}}

    function AR1(; constraint = nothing)
        return new(constraint)
    end
end

(::AR1)(args...) = error("AR1(...) functor is only intended for use inside @formula; not callable directly.")

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

"""
    BYM2(W; id_to_node = nothing, normalize_var = true, singleton_policy = :gaussian, additional_constraints = nothing)

Formula functor for BYM2 (Besag-York-Mollié with improved parameterization) random effects.

The BYM2 model combines spatial (ICAR) and unstructured (IID) random effects with
intuitive mixing and precision parameters. It is a reparameterization of the classic
BYM model that facilitates prior specification (Riebler et al. 2016).

# Arguments
- `W`: Adjacency matrix for the spatial structure
- `id_to_node`: Optional mapping from region identifiers to integer node indices (1-based)
- `normalize_var`: Whether to normalize variance (default: true, required for BYM2)
- `singleton_policy`: How to handle isolated nodes (`:gaussian` or `:degenerate`)
- `additional_constraints`: Optional additional constraints beyond built-in sum-to-zero

# Model Structure
The BYM2 model creates a 2n-dimensional latent field:
- Components 1:n are spatial effects (variance-normalized ICAR)
- Components (n+1):2n are unstructured effects (IID)
- In the linear predictor: η[i] = ... + u*[i] + v*[i]

# Hyperparameters
- `τ`: Overall precision (τ > 0)
- `φ`: Mixing parameter (0 < φ < 1), proportion of unstructured variance
  - φ = 0: pure spatial model
  - φ = 1: pure unstructured model
  - φ = 0.5: equal spatial and unstructured variance

# Usage
```julia
# Create BYM2 functor with adjacency matrix
W = adjacency_matrix
bym2 = BYM2(W)
@formula(y ~ 0 + bym2(region))

# With categorical region IDs
id_map = Dict("WesternIsles" => 11, "Highland" => 12, ...)
bym2_mapped = BYM2(W; id_to_node = id_map)
@formula(y ~ 0 + bym2_mapped(region))

# With custom singleton policy
bym2_deg = BYM2(W; singleton_policy = :degenerate)
@formula(y ~ 0 + bym2_deg(region))
```

# Notes
- BYM2 always uses variance normalization (normalize_var = true)
- Sum-to-zero constraint is automatically applied to the spatial component
- You must create a BYM2 instance before using it in a formula
- Calling the functor directly is unsupported outside formula parsing

# References
Riebler, A., Sørbye, S. H., Simpson, D., & Rue, H. (2016).
An intuitive Bayesian spatial model for disease mapping that accounts for scaling.
Statistical Methods in Medical Research, 25(4), 1145-1165.
"""
struct BYM2{WT <: AbstractMatrix, MT}
    W::WT
    id_to_node::MT  # may be Nothing or a mapping supporting getindex
    normalize_var::Bool
    singleton_policy::Symbol
    additional_constraints::Union{Nothing, Tuple{AbstractMatrix, AbstractVector}}

    function BYM2(
            W::WT;
            id_to_node = nothing,
            normalize_var::Bool = true,
            singleton_policy::Symbol = :gaussian,
            additional_constraints = nothing,
        ) where {WT}
        # BYM2 requires variance normalization
        if !normalize_var
            throw(ArgumentError("BYM2 requires variance normalization (normalize_var must be true)"))
        end
        return new{WT, typeof(id_to_node)}(W, id_to_node, normalize_var, singleton_policy, additional_constraints)
    end
end

(::BYM2)(args...) = error("BYM2(...) functor is only intended for use inside @formula; not callable directly.")
