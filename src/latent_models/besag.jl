using SparseArrays
using LinearAlgebra
using Distributions
using Graphs
using LinearSolve

export BesagModel

"""
    BesagModel(adjacency::AbstractMatrix; regularization::Float64 = 1e-5, normalize_var::Bool = true, singleton_policy::Symbol = :gaussian, alg=CHOLMODFactorization())

A Besag model for spatial latent effects on graphs using intrinsic Conditional Autoregressive (CAR) structure.

The Besag model represents spatial dependence where each node's precision depends on its graph neighbors.
This creates a graph Laplacian precision structure that's widely used for spatial smoothing on irregular lattices.

# Mathematical Description

For a graph with adjacency matrix W, the precision matrix follows:
- Q[i,j] = -τ if nodes i and j are neighbors (W[i,j] = 1)
- Q[i,i] = τ * degree[i] where degree[i] = sum(W[i,:])
- All other entries are 0

Since this matrix is singular (rank n-1), we handle it as an intrinsic GMRF by:
1. Scaling by τ first, then adding small regularization (1e-5) to diagonal for numerical stability
2. Adding sum-to-zero constraint: sum(x) = 0

# Hyperparameters
- `τ`: Precision parameter (τ > 0)

# Fields
- `adjacency::M`: Adjacency matrix W (preserves input structure - sparse, SymTridiagonal, etc.)
- `regularization::Float64`: Small value added to diagonal after scaling (default 1e-5)
- `alg::Alg`: LinearSolve algorithm for solving linear systems

# Example
```julia
# 4-node cycle graph - can use sparse, SymTridiagonal, or Matrix
W = sparse(Bool[0 1 0 1; 1 0 1 0; 0 1 0 1; 1 0 1 0])
model = BesagModel(W)
gmrf = model(τ=1.0)  # Returns ConstrainedGMRF with sum-to-zero constraint using CHOLMODFactorization

# Or specify custom algorithm
model = BesagModel(W, alg=LDLtFactorization())
gmrf = model(τ=1.0)
```
"""
struct BesagModel{M <: AbstractMatrix, NF, QT <: AbstractMatrix, PT, Alg, C} <: LatentModel
    adjacency::M
    regularization::Float64
    connected_components::Vector{Vector{Int}}
    normalization_factor::NF
    Q::QT
    singleton_policy::PT
    alg::Alg
    additional_constraints::C

    function BesagModel(
            adjacency::AbstractMatrix;
            regularization::Float64 = 1.0e-5,
            normalize_var = Val{true}(),
            singleton_policy = Val{:gaussian}(),
            alg = CHOLMODFactorization(),
            additional_constraints = nothing,
        )
        # Convert to appropriate sparse/structured format
        adj = if adjacency isa Matrix
            sparse(Bool.(adjacency))  # Convert Matrix to sparse for efficiency
        else
            adjacency  # Preserve structure (SymTridiagonal, SparseMatrixCSC, etc.)
        end

        # Validate adjacency matrix
        n = size(adj, 1)
        size(adj, 2) == n || throw(ArgumentError("Adjacency matrix must be square, got size $(size(adj))"))

        # Check symmetry
        adj == adj' || throw(ArgumentError("Adjacency matrix must be symmetric"))

        # Check diagonal is zero
        all(adj[i, i] == 0 for i in 1:n) || throw(ArgumentError("Adjacency matrix must have zero diagonal"))

        # Validate regularization
        regularization > 0 || throw(ArgumentError("Regularization must be positive, got $regularization"))
        # Validate singleton policy
        singleton_policy in (Val{:gaussian}(), Val{:degenerate}()) ||
            throw(ArgumentError("singleton_policy must be Val{:gaussian}() or Val{:degenerate}(), got $(singleton_policy)"))

        # Check for :sumtozero (redundant for Besag)
        if additional_constraints === :sumtozero
            throw(ArgumentError("BesagModel already includes sum-to-zero constraints by default. Use additional_constraints for extra constraints beyond sum-to-zero."))
        end

        # Process additional constraints using helper
        processed_additional = _process_constraint(additional_constraints, n)

        D = Diagonal(adj * ones(n))  # Degree matrix
        Q = (D - adj)                # Base intrinsic precision (per τ)

        G = SimpleGraph(adj)
        comps = connected_components(G)

        _enforce_singleton_policy_on_Q!(Q, comps, singleton_policy)

        normalization_factor = _compute_normalization(Q, comps, normalize_var, singleton_policy)
        return new{typeof(adj), typeof(normalization_factor), typeof(Q), typeof(singleton_policy), typeof(alg), typeof(processed_additional)}(adj, regularization, comps, normalization_factor, Q, singleton_policy, alg, processed_additional)
    end
end

function Base.length(model::BesagModel)
    return size(model.adjacency, 1)
end

function hyperparameters(model::BesagModel)
    return (τ = Real,)
end

function _validate_besag_parameters(; τ::Real)
    τ > 0 || throw(ArgumentError("Precision parameter τ must be positive, got τ=$τ"))
    return nothing
end

function _get_constraint_matrix(n, connected_comps, ::Val{:degenerate})
    n_constr = length(connected_comps)
    A = zeros(n_constr, n)
    for (i, comp) in enumerate(connected_comps)
        A[i, comp] .= 1.0
    end
    return A
end

function _get_constraint_matrix(n, connected_comps, ::Val{:gaussian})
    # No sum-to-zero constraint for singletons
    comps_no_singletons = filter(c -> length(c) > 1, connected_comps)
    n_constr = length(comps_no_singletons)
    A = zeros(n_constr, n)
    for (i, comp) in enumerate(comps_no_singletons)
        A[i, comp] .= 1.0
    end
    return A
end

function _enforce_singleton_policy_on_Q!(Q::AbstractMatrix, connected_comps, ::Val{:gaussian})
    for comp in connected_comps
        if length(comp) == 1
            idx = only(comp)
            Q[idx, idx] = 1.0  # proper Gaussian prior (scaled by τ later)
        end
    end
    return
end
_enforce_singleton_policy_on_Q!(::AbstractMatrix, connected_comps, ::Val{:degenerate}) = return

_geomean(x) = exp(mean(log.(x)))

function _compute_normalization(Q::AbstractMatrix, connected_comps, ::Val{true}, singleton_policy; regularization::Float64 = 1.0e-5)
    n = size(Q, 1)
    Qreg = Q + regularization * I
    x_tmp = GMRF(zeros(n), Qreg)
    A = _get_constraint_matrix(n, connected_comps, singleton_policy)
    e = zeros(size(A, 1))
    x_tmp_constr = ConstrainedGMRF(x_tmp, A, e)
    marginal_vars = var(x_tmp_constr)

    norms = ones(n)
    for comp in connected_comps
        if length(comp) > 1
            norms[comp] .= _geomean(marginal_vars[comp])
        end
    end
    return Diagonal(norms)
end
_compute_normalization(::AbstractMatrix, connected_comps, ::Val{false}, singleton_policy; kwargs...) = I

function precision_matrix(model::BesagModel; τ::Real, kwargs...)
    _validate_besag_parameters(; τ = τ)
    Q = model.normalization_factor * τ * model.Q  # Scale by τ first
    Q += model.regularization * I  # Add regularization
    return Q
end

function mean(model::BesagModel; kwargs...)
    n = size(model.adjacency, 1)
    return zeros(n)
end

function constraints(model::BesagModel; kwargs...)
    # Besag always has built-in constraints based on connected components and singleton policy
    n = size(model.adjacency, 1)
    A_builtin = _get_constraint_matrix(n, model.connected_components, model.singleton_policy)
    e_builtin = zeros(size(A_builtin, 1))

    # If no additional constraints, return just built-in
    if model.additional_constraints === nothing
        return (A_builtin, e_builtin)
    end

    # Otherwise stack constraints
    A_add, e_add = model.additional_constraints
    A_combined = vcat(A_builtin, A_add)
    e_combined = vcat(e_builtin, e_add)
    return (A_combined, e_combined)
end

function model_name(::BesagModel)
    return :besag
end

# The (model::LatentModel)(; kwargs...) method is inherited from the abstract type
