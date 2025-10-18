using SparseArrays
using LinearAlgebra
using LinearSolve

export CombinedModel

"""
    CombinedModel(components::Vector{<:LatentModel}; alg=CHOLMODFactorization())

A combination of multiple LatentModel instances into a single block-structured GMRF.

This enables modeling with multiple latent components, such as the popular BYM model
(Besag-York-Mollié) which combines spatial (Besag) and independent (IID) effects.

# Mathematical Description

Given k component models with sizes n₁, n₂, ..., nₖ:
- Combined precision matrix: Q = blockdiag(Q₁, Q₂, ..., Qₖ)
- Combined mean vector: μ = vcat(μ₁, μ₂, ..., μₖ)
- Combined constraints: Block-diagonal constraint structure preserving individual constraints

# Parameter Naming

To avoid conflicts when multiple models have the same hyperparameters (e.g., multiple τ),
parameters are automatically prefixed with model names:
- Single occurrence: τ_besag, ρ_ar1
- Multiple occurrences: τ_besag, τ_besag_2, τ_besag_3

# Fields
- `components::Vector{<:LatentModel}`: The individual latent models
- `component_sizes::Vector{Int}`: Cached sizes of each component
- `total_size::Int`: Total size of the combined model
- `alg::Alg`: LinearSolve algorithm for solving linear systems

# Example - BYM Model
```julia
# BYM model: spatial Besag + independent IID effects
W = sparse(adjacency_matrix)  # Spatial adjacency
besag = BesagModel(W)         # Spatial component
iid = IIDModel(n)            # Independent component

# Vector constructor
bym = CombinedModel([besag, iid])
# Or variadic constructor (syntactic sugar)
bym = CombinedModel(besag, iid)

# With custom algorithm
bym = CombinedModel([besag, iid], alg=LDLtFactorization())

# Usage with automatically prefixed parameters
gmrf = bym(τ_besag=1.0, τ_iid=2.0)
```
"""
struct CombinedModel{Alg} <: LatentModel
    components::Vector{<:LatentModel}
    component_sizes::Vector{Int}
    total_size::Int
    alg::Alg

    function CombinedModel{Alg}(components::Vector{<:LatentModel}, alg::Alg) where {Alg}
        isempty(components) && throw(ArgumentError("Cannot create CombinedModel with empty components"))

        # Cache component sizes for efficiency
        sizes = [length(comp) for comp in components]
        total = sum(sizes)

        return new{Alg}(components, sizes, total, alg)
    end
end

function CombinedModel(components::Vector{<:LatentModel}; alg = CHOLMODFactorization())
    return CombinedModel{typeof(alg)}(components, alg)
end

# Convenience constructor for variadic arguments
function CombinedModel(components::LatentModel...; alg = CHOLMODFactorization())
    return CombinedModel(collect(components); alg = alg)
end

function Base.length(model::CombinedModel)
    return model.total_size
end

function hyperparameters(model::CombinedModel)
    result = NamedTuple()
    name_counts = Dict{Symbol, Int}()  # Track how many times each name appears

    for component in model.components
        base_name = model_name(component)

        # Handle duplicates: first occurrence gets no suffix, then _2, _3, etc.
        name_counts[base_name] = get(name_counts, base_name, 0) + 1
        suffix = name_counts[base_name] == 1 ? "" : "_$(name_counts[base_name])"
        final_name = Symbol("$(base_name)$(suffix)")

        # Add parameters with this final name as prefix
        component_params = hyperparameters(component)
        for (param, type) in pairs(component_params)
            prefixed = Symbol("$(param)_$(final_name)")
            result = merge(result, (prefixed => type,))
        end
    end
    return result
end

# Helper function to extract parameters for a specific component
function _extract_component_params(component::LatentModel, component_idx::Int, model::CombinedModel, kwargs...)
    base_name = model_name(component)

    # Convert kwargs to dictionary for easier access
    kwargs_dict = Dict{Symbol, Any}(kwargs...)

    # Determine the suffix for this component
    name_counts = Dict{Symbol, Int}()
    suffix = ""
    for i in 1:component_idx
        comp_name = model_name(model.components[i])
        name_counts[comp_name] = get(name_counts, comp_name, 0) + 1
        if i == component_idx
            suffix = name_counts[comp_name] == 1 ? "" : "_$(name_counts[comp_name])"
        end
    end
    final_name = Symbol("$(base_name)$(suffix)")

    # Extract parameters that belong to this component
    component_params = Dict{Symbol, Any}()
    component_param_names = keys(hyperparameters(component))

    for param_name in component_param_names
        prefixed_name = Symbol("$(param_name)_$(final_name)")
        if haskey(kwargs_dict, prefixed_name)
            component_params[param_name] = kwargs_dict[prefixed_name]
        else
            throw(ArgumentError("Missing required parameter: $prefixed_name"))
        end
    end

    return component_params
end

function precision_matrix(model::CombinedModel; kwargs...)
    # Build block diagonal precision matrix
    component_matrices = []

    for (i, component) in enumerate(model.components)
        comp_params = _extract_component_params(component, i, model, kwargs...)
        Q_comp = precision_matrix(component; comp_params...)
        push!(component_matrices, Q_comp)
    end

    return _blockdiag(component_matrices...)
end

function mean(model::CombinedModel; kwargs...)
    # Build stacked mean vector
    component_means = []

    for (i, component) in enumerate(model.components)
        comp_params = _extract_component_params(component, i, model, kwargs...)
        μ_comp = mean(component; comp_params...)
        push!(component_means, μ_comp)
    end

    return vcat(component_means...)
end

function constraints(model::CombinedModel; kwargs...)
    # Collect constraints from all components
    component_constraints = []
    constrained_components = Int[]

    for (i, component) in enumerate(model.components)
        comp_params = _extract_component_params(component, i, model, kwargs...)
        constraint_info = constraints(component; comp_params...)

        if constraint_info !== nothing
            push!(component_constraints, constraint_info)
            push!(constrained_components, i)
        end
    end

    # If no constraints, return nothing
    isempty(component_constraints) && return nothing

    # Build combined constraint matrices
    total_constraints = sum(size(info[1], 1) for info in component_constraints)
    A_combined = zeros(total_constraints, model.total_size)
    e_combined = zeros(total_constraints)

    constraint_row = 1
    for (constraint_idx, component_idx) in enumerate(constrained_components)
        A_comp, e_comp = component_constraints[constraint_idx]
        n_constraints = size(A_comp, 1)

        # Calculate column range for this component
        col_start = sum(model.component_sizes[1:(component_idx - 1)]) + 1
        col_end = sum(model.component_sizes[1:component_idx])

        # Place component constraints in the right block
        A_combined[constraint_row:(constraint_row + n_constraints - 1), col_start:col_end] = A_comp
        e_combined[constraint_row:(constraint_row + n_constraints - 1)] = e_comp

        constraint_row += n_constraints
    end

    return (A_combined, e_combined)
end

function model_name(::CombinedModel)
    return :combined
end

# Helper to ensure matrix is sparse
_ensure_sparse(A::SparseMatrixCSC) = A
_ensure_sparse(A::AbstractMatrix) = sparse(A)

# Helper function for block diagonal construction - always returns sparse
function _blockdiag(matrices...)
    if isempty(matrices)
        throw(ArgumentError("Cannot create block diagonal of empty matrix list"))
    end

    # Convert all to sparse and concatenate
    sparse_matrices = [_ensure_sparse(M) for M in matrices]

    # Calculate offsets
    row_offsets = cumsum([0; [size(M, 1) for M in sparse_matrices[1:(end - 1)]]])
    col_offsets = cumsum([0; [size(M, 2) for M in sparse_matrices[1:(end - 1)]]])

    # Collect all entries
    I_vals = Int[]
    J_vals = Int[]
    V_vals = Float64[]

    for (i, M) in enumerate(sparse_matrices)
        rows_M, cols_M, vals_M = findnz(M)
        append!(I_vals, rows_M .+ row_offsets[i])
        append!(J_vals, cols_M .+ col_offsets[i])
        append!(V_vals, vals_M)
    end

    total_rows = sum(size(M, 1) for M in sparse_matrices)
    total_cols = sum(size(M, 2) for M in sparse_matrices)

    return sparse(I_vals, J_vals, V_vals, total_rows, total_cols)
end

# The (model::LatentModel)(; kwargs...) method is inherited from the abstract type

# COV_EXCL_START
function Base.show(io::IO, model::CombinedModel)
    k = length(model.components)
    print(io, "CombinedModel with $(k) component$(k == 1 ? "" : "s") (total=$(model.total_size)):")
    offset = 0
    for (i, comp) in enumerate(model.components)
        sz = model.component_sizes[i]
        pname = model_name(comp)
        params = join(string.(collect(keys(hyperparameters(comp)))), ", ")
        print(io, "\n  [$i] $(pname) (n=$(sz))")
        if !isempty(params)
            print(io, ", hyperparameters = [", params, "]")
        end
        offset += sz
    end
    return
end
# COV_EXCL_STOP
