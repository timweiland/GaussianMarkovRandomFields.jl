using SparseArrays
using LinearAlgebra

export BesagModel

"""
    BesagModel(adjacency::AbstractMatrix; regularization::Float64 = 1e-5)

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

# Example
```julia
# 4-node cycle graph - can use sparse, SymTridiagonal, or Matrix
W = sparse(Bool[0 1 0 1; 1 0 1 0; 0 1 0 1; 1 0 1 0])
model = BesagModel(W)
gmrf = model(τ=1.0)  # Returns ConstrainedGMRF with sum-to-zero constraint
```
"""
struct BesagModel{M <: AbstractMatrix} <: LatentModel
    adjacency::M
    regularization::Float64

    function BesagModel(adjacency::AbstractMatrix; regularization::Float64 = 1.0e-5)
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

        # Check for isolated nodes (degree = 0)
        degrees = vec(sum(adj, dims = 2))
        all(degrees .> 0) || throw(ArgumentError("Graph cannot have isolated nodes (nodes with degree 0)"))

        # Validate regularization
        regularization > 0 || throw(ArgumentError("Regularization must be positive, got $regularization"))

        return new{typeof(adj)}(adj, regularization)
    end
end

function hyperparameters(model::BesagModel)
    return (τ = Real,)
end

function _validate_besag_parameters(; τ::Real)
    τ > 0 || throw(ArgumentError("Precision parameter τ must be positive, got τ=$τ"))
    return nothing
end

function precision_matrix(model::BesagModel; τ::Real)
    _validate_besag_parameters(; τ = τ)

    W = model.adjacency
    n = size(W, 1)

    # Compute graph Laplacian: D - W where D is degree matrix
    D = Diagonal(W * ones(n))  # Degree matrix
    Q = τ * (D - W)            # Scale by τ first
    Q += model.regularization * I  # Add regularization

    return Q
end

function mean(model::BesagModel; kwargs...)
    n = size(model.adjacency, 1)
    return zeros(n)
end

function constraints(model::BesagModel; kwargs...)
    # Sum-to-zero constraint: sum(x) = 0
    # A is 1×n matrix of all ones, e is [0]
    n = size(model.adjacency, 1)
    A = ones(1, n)  # 1×n matrix
    e = [0.0]       # Constraint vector
    return (A, e)
end

# The (model::LatentModel)(; kwargs...) method is inherited from the abstract type
