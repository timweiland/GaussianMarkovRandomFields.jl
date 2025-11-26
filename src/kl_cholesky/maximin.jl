using NearestNeighbors, DataStructures
using SparseArrays

export reverse_maximin_ordering, reverse_maximin_ordering_and_sparsity_pattern, sparsity_pattern_from_ordering, form_supernodes

"""
    reverse_maximin_ordering(X::AbstractMatrix; point_tree = KDTree(X))

Compute a reverse maximin ordering of spatial points.

The reverse maximin ordering selects points greedily to maximize the distance to
previously selected points. This creates a hierarchical structure where similar
points appear consecutively in the ordering, which is beneficial for sparse
Cholesky factorization.

# Arguments
- `X::AbstractMatrix`: Input point locations (d × n matrix where d is spatial dimension)

# Keyword Arguments
- `point_tree = KDTree(X)`: Pre-computed KDTree for efficient nearest neighbor queries

# Returns
- `P::Vector{Int}`: Permutation vector (ordering of points)
- `ℓ::Vector{Float64}`: Maximin distances for each point

# Details
The algorithm maintains a priority queue of unselected points, ordered by their
distance to the nearest selected point. At each step, it selects the point furthest
from all previously selected points.

# See also
[`reverse_maximin_ordering_and_sparsity_pattern`](@ref), [`sparse_approximate_cholesky`](@ref)
"""
function reverse_maximin_ordering(X::AbstractMatrix; point_tree = KDTree(X))
    N = size(X, 2)
    P = zeros(Int, N)
    ℓ = zeros(Float64, N)
    dists = fill(floatmax(Float64), N)
    heap = MutableBinaryHeap{Float64, DataStructures.FasterReverse}(dists)
    neighbors = Int[]
    selected = falses(N)
    for k in N:-1:1
        ℓ_cur, idx_cur = top_with_handle(heap)
        pop!(heap)
        ℓ[idx_cur] = ℓ_cur
        P[k] = idx_cur
        selected[idx_cur] = true

        x_cur = @view(X[:, idx_cur])
        empty!(neighbors)
        inrange!(neighbors, point_tree, x_cur, ℓ_cur)
        for neighbor_idx in neighbors
            neighbor_idx == idx_cur && continue
            selected[neighbor_idx] && continue

            x_neighbor = @view(X[:, neighbor_idx])
            dist_neighbor_cur = point_tree.metric(x_cur, x_neighbor)
            if dists[neighbor_idx] > dist_neighbor_cur
                dists[neighbor_idx] = dist_neighbor_cur
                DataStructures.update!(heap, neighbor_idx, dist_neighbor_cur)
            end
        end
    end

    return P, ℓ
end

"""
    reverse_maximin_ordering_and_sparsity_pattern(X::AbstractMatrix, ρ::Real; lower = true)

Compute reverse maximin ordering and determine the sparsity pattern for sparse Cholesky factorization.

This function combines the reverse maximin ordering with sparsity pattern construction
based on spatial neighborhoods. The resulting sparsity pattern determines which entries
of the Cholesky factor will be nonzero.

# Arguments
- `X::AbstractMatrix`: Input point locations (d × n matrix where d is spatial dimension)
- `ρ::Real`: Neighborhood radius multiplier for sparsity pattern construction

# Keyword Arguments
- `lower::Bool = true`: If true, return lower triangular pattern; otherwise upper triangular

# Returns
- `S::SparseMatrixCSC`: Sparsity pattern matrix (all nonzeros are 0.0, structure only)
- `P::Vector{Int}`: Permutation vector from reverse maximin ordering
- `ℓ::Vector{Float64}`: Maximin distances for each point

# Details
For each point i, the algorithm includes entries in the sparsity pattern for all neighbors
within distance `ρ × ℓᵢ`, where `ℓᵢ` is the maximin distance for point i. Larger `ρ`
values create denser (and more accurate) approximations.

# See also
[`reverse_maximin_ordering`](@ref), [`sparse_approximate_cholesky`](@ref)
"""
function reverse_maximin_ordering_and_sparsity_pattern(X::AbstractMatrix, ρ::Real; lower = true)
    point_tree = KDTree(X)
    P, ℓ = reverse_maximin_ordering(X; point_tree)
    S = sparsity_pattern_from_ordering(X, P, ℓ, ρ; lower, point_tree)
    return S, P, ℓ
end

"""
    sparsity_pattern_from_ordering(X::AbstractMatrix, P::Vector{Int}, ℓ::Vector{Float64}, ρ::Real; lower = true, point_tree = KDTree(X))

Construct sparsity pattern for sparse Cholesky factorization from a pre-computed ordering.

This function allows users who already have a permutation `P` and maximin distances `ℓ`
(from `reverse_maximin_ordering` or other sources) to construct the sparsity pattern
independently. This is useful when the ordering has been computed separately or when
experimenting with different sparsity parameters `ρ` on the same ordering.

# Arguments
- `X::AbstractMatrix`: Input point locations (d × n matrix where d is spatial dimension)
- `P::Vector{Int}`: Permutation vector (ordering of points)
- `ℓ::Vector{Float64}`: Maximin distances for each point
- `ρ::Real`: Neighborhood radius multiplier for sparsity pattern construction

# Keyword Arguments
- `lower::Bool = true`: If true, return lower triangular pattern; otherwise upper triangular
- `point_tree = KDTree(X)`: Pre-computed KDTree for efficient nearest neighbor queries

# Returns
- `S::SparseMatrixCSC`: Sparsity pattern matrix (all nonzeros are 0.0, structure only)

# Details
For each point i, the algorithm includes entries in the sparsity pattern for all neighbors
within distance `ρ × ℓᵢ`, where `ℓᵢ` is the maximin distance for point i. Larger `ρ`
values create denser (and more accurate) approximations.

# See also
[`reverse_maximin_ordering`](@ref), [`reverse_maximin_ordering_and_sparsity_pattern`](@ref), [`sparse_approximate_cholesky`](@ref)
"""
function sparsity_pattern_from_ordering(
        X::AbstractMatrix,
        P::Vector{Int},
        ℓ::Vector{Float64},
        ρ::Real;
        lower::Bool = true,
        point_tree = KDTree(X)
    )
    P_inv = invperm(P)
    Is = Int[]
    Js = Int[]
    neighbors = Int[]
    for j in eachindex(P)
        cur_point_idx = P[j]
        x_cur = @view(X[:, cur_point_idx])
        ℓ_cur = ℓ[cur_point_idx]
        empty!(neighbors)
        inrange!(neighbors, point_tree, x_cur, ρ * ℓ_cur)
        for neighbor_point_idx in neighbors
            i = P_inv[neighbor_point_idx]
            i < j && continue
            push!(Is, i)
            push!(Js, j)
        end
    end
    N = length(P)
    if lower
        return spzeros(Is, Js, N, N)
    else
        return spzeros(Js, Is, N, N)
    end
end
