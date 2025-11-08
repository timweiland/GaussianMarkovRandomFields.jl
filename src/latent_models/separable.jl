using LinearAlgebra
using SparseArrays
using LinearMaps

export SeparableModel

"""
    SeparableModel{T<:Tuple{Vararg{LatentModel}}, Alg} <: LatentModel

Represents a separable (Kronecker product) latent model for multi-dimensional processes.

The precision matrix is the Kronecker product of the component precision matrices:
```
Q = Q_1 ⊗ Q_2 ⊗ ... ⊗ Q_N
```
where the rightmost component varies fastest in the vectorized representation.

For example, `SeparableModel(temporal, spatial)` produces `Q_time ⊗ Q_space`, yielding
a block-tridiagonal structure when Q_time is tridiagonal (e.g., RW1).

# Fields
- `components::T`: Tuple of LatentModel components
- `alg`: Linear solver algorithm

# Example
```julia
# Space-time separable model (standalone)
temporal = RW1Model(n_time)
spatial = BesagModel(adjacency_matrix)
st_model = SeparableModel(temporal, spatial)  # Q = Q_time ⊗ Q_space

# Instantiate with hyperparameters
# Standalone: τ_rw1, τ_besag
# In CombinedModel: τ_rw1_separable, τ_besag_separable
gmrf = st_model(τ_rw1=1.0, τ_besag=2.0)
```

# Notes
- Requires at least 2 components
- Component ordering: rightmost component varies fastest (e.g., space in time×space model)
- Follows R-INLA convention: Q = Q_group ⊗ Q_main
- Hyperparameters are suffixed with component model names (e.g., `τ_rw1`, `τ_besag`)
- When used in CombinedModel, an additional `_separable` suffix is added by the parent
- Constraints from each component are composed via Kronecker products with identity matrices
- Redundant constraints are automatically removed to ensure full row rank
"""
struct SeparableModel{T <: Tuple{Vararg{LatentModel}}, Alg} <: LatentModel
    components::T
    alg::Alg
end

# Constructor
function SeparableModel(components::LatentModel...; alg = nothing)
    length(components) >= 2 ||
        error("SeparableModel requires at least 2 components, got $(length(components))")
    return SeparableModel(components, alg)
end

# LatentModel interface implementation

function Base.length(model::SeparableModel)
    return prod(length(c) for c in model.components)
end

model_name(model::SeparableModel) = :separable

function hyperparameters(model::SeparableModel)
    # Collect all component hyperparameters
    all_params = NamedTuple[]

    # Track counts of each model name for indexing duplicates
    name_counts = Dict{Symbol, Int}()

    for component in model.components
        comp_params = hyperparameters(component)
        comp_model_name = model_name(component)

        # Increment count for this model type
        count = get(name_counts, comp_model_name, 0) + 1
        name_counts[comp_model_name] = count

        # Create suffixed parameters (no _separable_ prefix here)
        suffix = count == 1 ? "" : "_$count"
        suffix_keys = [Symbol("$(k)_$(comp_model_name)$(suffix)") for k in keys(comp_params)]
        suffixed_params = NamedTuple{Tuple(suffix_keys)}(values(comp_params))

        push!(all_params, suffixed_params)
    end

    # Merge all parameter tuples
    return merge(all_params...)
end

function mean(model::SeparableModel; kwargs...)
    # Extract hyperparameters for each component
    comp_kwargs = _extract_component_kwargs(model, kwargs)

    # Compute means for each component
    means = [mean(comp; comp_kwargs[i]...) for (i, comp) in enumerate(model.components)]

    # If all components have zero mean, return zero vector
    if all(iszero, means)
        return zeros(length(model))
    end

    # Otherwise, compute Kronecker product of means
    # For SeparableModel([comp1, comp2]), we compute: kron(mean1, mean2)
    # This vectorizes with comp2 (rightmost) varying fastest
    return foldl(kron, means)
end

function precision_matrix(model::SeparableModel; kwargs...)
    # Extract hyperparameters for each component
    comp_kwargs = _extract_component_kwargs(model, kwargs)

    # Compute precision matrices for each component
    Qs = [precision_matrix(comp; comp_kwargs[i]...) for (i, comp) in enumerate(model.components)]

    # Return Kronecker product using LinearMaps
    # For SeparableModel([comp1, comp2]), we compute: Q1 ⊗ Q2
    # This matches the mean vectorization (comp2 varying fastest)
    return foldl(kron, Qs)
end

"""
    _remove_redundant_constraints(A::AbstractMatrix, e::AbstractVector; tol=1e-10)

Remove linearly dependent rows from the constraint matrix A and corresponding entries from e.
Returns the reduced constraint system (A_reduced, e_reduced).
"""
function _remove_redundant_constraints(A::AbstractMatrix, e::AbstractVector; tol = 1.0e-10)
    A_dense = Matrix(A)
    m, n = size(A_dense)

    # QR decomposition with column pivoting to find independent rows
    # We work with A' to find independent columns of A', which are independent rows of A
    F = qr(A_dense', ColumnNorm())

    # Determine rank
    R_diag = abs.(diag(F.R))
    rank_A = sum(R_diag .> tol * maximum(R_diag))

    if rank_A == m
        # Already full rank
        return (A, e)
    end

    # Keep only the first rank_A rows (they correspond to independent rows of A)
    # The permutation F.p tells us which rows to keep
    rows_to_keep = sort(F.p[1:rank_A])

    A_reduced = A[rows_to_keep, :]
    e_reduced = e[rows_to_keep]

    return (A_reduced, e_reduced)
end

function constraints(model::SeparableModel; kwargs...)
    # Extract hyperparameters for each component
    comp_kwargs = _extract_component_kwargs(model, kwargs)

    # Collect constraint information from each component
    constraint_list = Tuple{AbstractMatrix, AbstractVector}[]

    for (i, comp) in enumerate(model.components)
        comp_constraint = constraints(comp; comp_kwargs[i]...)

        if comp_constraint !== nothing
            A_i, e_i = comp_constraint

            # Compute sizes before and after this component
            n_before = i == 1 ? 1 : prod(length(model.components[j]) for j in 1:(i - 1))
            n_after = i == length(model.components) ? 1 : prod(length(model.components[j]) for j in (i + 1):length(model.components))

            # Expand constraint to full dimensionality via Kronecker products
            # For Q = Q_1 ⊗ Q_2 ⊗ ... ⊗ Q_N, constraint on component i is:
            # A_full = I_before ⊗ A_i ⊗ I_after
            # where I_before has size n_before and I_after has size n_after

            if n_before == 1 && n_after == 1
                # No expansion needed
                push!(constraint_list, (A_i, e_i))
            else
                # Use LinearMaps for lazy Kronecker products
                I_before = LinearAlgebra.I(n_before)
                I_after = LinearAlgebra.I(n_after)

                # Kronecker product: I_before ⊗ A_i ⊗ I_after
                A_full = kron(kron(I_before, A_i), I_after)

                # Expand constraint vector
                e_i_expanded = vec(e_i)
                n_constraints = size(A_i, 1)
                e_full = zeros(n_constraints * n_before * n_after)

                for idx_after in 1:n_after
                    for idx_constraint in 1:n_constraints
                        for idx_before in 1:n_before
                            # Linear index in the expanded vector
                            idx = idx_before + (idx_constraint - 1) * n_before + (idx_after - 1) * n_before * n_constraints
                            e_full[idx] = e_i_expanded[idx_constraint]
                        end
                    end
                end

                # Convert to concrete sparse matrix if A_full is a LinearMap
                if A_full isa LinearMaps.LinearMap
                    A_full_sparse = sparse(to_matrix(A_full))
                else
                    A_full_sparse = sparse(A_full)
                end

                push!(constraint_list, (A_full_sparse, e_full))
            end
        end
    end

    # If no constraints, return nothing
    if isempty(constraint_list)
        return nothing
    end

    # Stack all constraints vertically
    A_combined = vcat([A for (A, _) in constraint_list]...)
    e_combined = vcat([e for (_, e) in constraint_list]...)

    # Remove redundant constraints to ensure full row rank
    return _remove_redundant_constraints(A_combined, e_combined)
end

"""
    _extract_component_kwargs(model::SeparableModel, kwargs)

Extract hyperparameters for each component from the combined kwargs.

Returns a vector of NamedTuples, one per component.
"""
function _extract_component_kwargs(model::SeparableModel, kwargs)
    # Track counts of each model name
    name_counts = Dict{Symbol, Int}()

    comp_kwargs = NamedTuple[]

    for component in model.components
        comp_model_name = model_name(component)
        comp_params_template = hyperparameters(component)

        # Increment count for this model type
        count = get(name_counts, comp_model_name, 0) + 1
        name_counts[comp_model_name] = count

        # Extract parameters for this component
        suffix = count == 1 ? "" : "_$count"
        comp_kw = NamedTuple()

        for param_name in keys(comp_params_template)
            # Look for {param}_{modelname}{suffix} in kwargs
            full_key = Symbol("$(param_name)_$(comp_model_name)$(suffix)")

            if haskey(kwargs, full_key)
                comp_kw = merge(comp_kw, NamedTuple{(param_name,)}((kwargs[full_key],)))
            end
        end

        push!(comp_kwargs, comp_kw)
    end

    return comp_kwargs
end
