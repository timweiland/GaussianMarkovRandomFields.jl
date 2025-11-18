import Base.Reverse

export SupernodeClustering, form_supernodes

"""
    SupernodeClustering

Represents a clustering of matrix columns into supernodes for sparse Cholesky factorization.

# Fields
- `column_indices::Vector{Vector{Int}}`: For each supernode, the column indices belonging to it
- `row_indices::Vector{SortedSet{Int}}`: For each supernode, the row indices in its sparsity pattern

Supernodes group columns with similar sparsity patterns to enable more efficient
factorization through larger dense linear algebra operations.
"""
struct SupernodeClustering{TC, TR}
    column_indices::TC
    row_indices::TR
end

# Number of supernodes
length(sc::SupernodeClustering) = length(sc.column_indices)

"""
    form_supernodes(S::SparseMatrixCSC, P, ℓ; λ = 1.5)

Cluster columns of a sparse matrix into supernodes based on maximin distances.

Groups columns with similar sparsity patterns and nearby maximin distances into
"supernodes" for more efficient sparse Cholesky factorization.

# Arguments
- `S::SparseMatrixCSC`: Sparse sparsity pattern matrix
- `P::Vector{Int}`: Permutation vector from reverse maximin ordering
- `ℓ::Vector{Float64}`: Maximin distances for each point

# Keyword Arguments
- `λ::Real = 1.5`: Clustering threshold. Columns are grouped if their maximin distances
  differ by less than this factor. Larger values create larger supernodes.

# Returns
- `SupernodeClustering`: The supernodal clustering structure

# Details
The algorithm processes columns sequentially in the permuted order. For each unassigned
column, it creates a new supernode and assigns all unassigned child columns (in the
sparsity pattern) whose maximin distances satisfy `ℓ[i] <= λ * ℓ[j]`.

# See also
[`SupernodeClustering`](@ref), [`sparse_approximate_cholesky`](@ref)
"""
function form_supernodes(S::SparseMatrixCSC, P, ℓ; λ = 1.5)
    column_indices = Vector{Int}[]
    row_indices = SortedSet{Int}[]
    N = size(S, 2)
    supernode_id = zeros(Int, N)
    N_supernode = 1

    for j in 1:N
        row_idcs = S.rowval[nzrange(S, j)]
        point_idx_j = P[j]

        cur_supernode_id = supernode_id[j]
        if cur_supernode_id == 0
            # No supernode assigned yet, create new one
            cur_columns = Int[]
            cur_rows = SortedSet{Int}(Base.Reverse) # reverse order!

            for i in row_idcs
                # Note that this includes j
                point_idx_i = P[i]
                if (supernode_id[i] == 0) && (ℓ[point_idx_i] <= λ * ℓ[point_idx_j])
                    # Add to supernode
                    push!(cur_columns, i)
                    supernode_id[i] = N_supernode
                end
                push!(cur_rows, i)
            end
            push!(column_indices, cur_columns)
            push!(row_indices, cur_rows)
            N_supernode += 1
        else
            # Already has a supernode assigned
            # Add any new children to the supernode
            for i in row_idcs
                if supernode_id[i] != cur_supernode_id
                    # Child
                    push!(row_indices[cur_supernode_id], i)
                end
            end
        end
    end
    return SupernodeClustering(column_indices, row_indices)
end
