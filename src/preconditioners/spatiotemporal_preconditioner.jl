using SparseArrays

export temporal_block_gauss_seidel

function temporal_block_gauss_seidel(A::SparseMatrixCSC, block_size)
    diag_blocks, off_diag_blocks = _extract_blocks(A, block_size)
    return TridiagonalBlockGaussSeidelPreconditioner(off_diag_blocks, diag_blocks)
end

function _extract_blocks(A::SparseMatrixCSC, block_size)
    I, J, V = findnz(A)
    return _extract_blocks(I, J, V, block_size)
end

function _extract_blocks(I::Vector{Int}, J::Vector{Int}, V::Vector, block_size)
    p = sortperm(I)
    I = I[p]
    J = J[p]
    V = V[p]

    # Initialize new I, J, V for the smaller matrix
    I_diag_block = Int[]
    J_diag_block = Int[]
    V_diag_block = eltype(V)[]  # Keep the element type of V for the new values

    I_off_diag_block = Int[]
    J_off_diag_block = Int[]
    V_off_diag_block = eltype(V)[]  # Keep the element type of V for the new values

    diag_blocks = []
    off_diag_blocks = []

    cur_diag_block_range = 1:block_size
    cur_off_diag_block_range = (1-block_size):0

    for idx in eachindex(I)
        i = I[idx]
        while i > cur_diag_block_range[end]
            push!(diag_blocks, (I_diag_block, J_diag_block, V_diag_block))
            if cur_off_diag_block_range[1] > 0
                push!(
                    off_diag_blocks,
                    (I_off_diag_block, J_off_diag_block, V_off_diag_block),
                )
            end
            I_diag_block = Int[]
            J_diag_block = Int[]
            V_diag_block = eltype(V)[]  # Keep the element type of V for the new values

            I_off_diag_block = Int[]
            J_off_diag_block = Int[]
            V_off_diag_block = eltype(V)[]  # Keep the element type of V for the new values

            cur_diag_block_range = cur_diag_block_range .+ block_size
            cur_off_diag_block_range = cur_off_diag_block_range .+ block_size
        end
        j = J[idx]

        # Check if the entry (i, j) falls within the block range
        if i in cur_diag_block_range && j in cur_diag_block_range
            push!(I_diag_block, i - first(cur_diag_block_range) + 1)  # Adjust row index to new block
            push!(J_diag_block, j - first(cur_diag_block_range) + 1)  # Adjust column index to new block
            push!(V_diag_block, V[idx])  # Keep the value as is
        end

        if i in cur_diag_block_range && j in cur_off_diag_block_range
            push!(I_off_diag_block, i - first(cur_diag_block_range) + 1)  # Adjust row index to new block
            push!(J_off_diag_block, j - first(cur_off_diag_block_range) + 1)  # Adjust column index to new block
            push!(V_off_diag_block, V[idx])  # Keep the value as is
        end
    end
    push!(diag_blocks, (I_diag_block, J_diag_block, V_diag_block))
    if cur_off_diag_block_range[1] > 0
        push!(off_diag_blocks, (I_off_diag_block, J_off_diag_block, V_off_diag_block))
    end

    diag_blocks = [
        sparse(I_block, J_block, V_block, block_size, block_size) for
        (I_block, J_block, V_block) in diag_blocks
    ]
    off_diag_blocks = [
        sparse(I_block, J_block, V_block, block_size, block_size) for
        (I_block, J_block, V_block) in off_diag_blocks
    ]
    return diag_blocks, off_diag_blocks
end
