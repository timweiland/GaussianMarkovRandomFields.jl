using AMD
using Random
using SelectedInversion
using LinearAlgebra

export RBMCStrategy, BlockRBMCStrategy

"""
    RBMCStrategy(n_samples; rng)

Rao-Blackwellized Monte Carlo estimator of a GMRF's marginal variances
based on [Siden2018](@cite).
Particularly useful in large-scale regimes where Takahashi recursions may be
too expensive.

# Arguments
- `n_samples::Int`: Number of samples to draw.

# Keyword arguments
- `rng::Random.AbstractRNG = Random.default_rng()`: Random number generator.
"""
struct RBMCStrategy
    n_samples::Int
    rng::Random.AbstractRNG

    function RBMCStrategy(n_samples::Int; rng::Random.AbstractRNG = Random.default_rng())
        new(n_samples, rng)
    end
end

"""
    BlockRBMCStrategy(n_samples; rng, enclosure_size)

Block Rao-Blackwellized Monte Carlo estimator of a GMRF's marginal variances
based on [Siden2018](@cite).
Achieves faster convergence than plain RBMC by considering blocks of nodes
rather than individual nodes, thus integrating more information about the
precision matrix.
`enclosure_size` specifies the size of these blocks. Larger values lead to
faster convergence (in terms of the number of samples) at the cost of
increased compute.
Thus, one should aim for a sweet spot between sampling costs and block operation
costs.

# Arguments
- `n_samples::Int`: Number of samples to draw.

# Keyword arguments
- `rng::Random.AbstractRNG = Random.default_rng()`: Random number generator.
- `enclosure_size::Int = 1`: Size of the blocks.
"""
struct BlockRBMCStrategy
    n_samples::Int
    rng::Random.AbstractRNG
    enclosure_size::Int

    function BlockRBMCStrategy(
        n_samples::Int;
        rng::Random.AbstractRNG = Random.default_rng(),
        enclosure_size::Int = 1,
    )
        new(n_samples, rng, enclosure_size)
    end
end

"""
    var(gmrf::GMRF, strategy::RBMCStrategy)

Compute marginal variances using Rao-Blackwellized Monte Carlo.
"""
function var(gmrf::AbstractGMRF, strategy::RBMCStrategy)
    Q = precision_matrix(gmrf)
    D = Array(diag(Q))
    D⁻¹ = 1 ./ D

    samples = [zeros(size(Q, 1)) for _ = 1:strategy.n_samples]
    for i = 1:strategy.n_samples
        rand!(strategy.rng, gmrf, samples[i])
        # Remove mean to get centered samples
        samples[i] .-= mean(gmrf)
    end
    sample_mat = hcat(samples...)

    # Q̃ = Q - Diagonal(D)
    transformed_samples = D⁻¹ .* (Q * sample_mat - D .* sample_mat)
    return D⁻¹ + reshape(var(transformed_samples, dims = 2), length(D))
end

# Helper functions for BlockRBMC
_all_neighbors(Q, i) = findnz(Q[i, :])[1]
_enclosure(Q, idcs) = vcat([_all_neighbors(Q, i) for i in idcs]...)

function _build_enclosure_idcs(Q, interior, enclosure_size = 1)
    enclosure_idcs = Int64[]
    new_idcs = copy(interior)
    explored = Set(interior)
    for _ = 1:enclosure_size
        new_idcs = Set(_enclosure(Q, new_idcs))
        new_idcs = setdiff(new_idcs, explored)
        append!(enclosure_idcs, collect(new_idcs))
        explored = explored ∪ new_idcs
    end
    return enclosure_idcs
end

function _build_disjoint_subsets(Q)
    visited = zeros(Bool, size(Q, 1))
    subsets = []
    for i = 1:size(Q, 1)
        if !visited[i]
            idcs = findnz(Q[i, :])[1]
            visited[idcs] .= true
            push!(subsets, idcs)
        end
    end
    return subsets
end

"""
    var(gmrf::GMRF, strategy::BlockRBMCStrategy)

Compute marginal variances using Block Rao-Blackwellized Monte Carlo.
"""
function var(gmrf::AbstractGMRF, strategy::BlockRBMCStrategy)
    Q = precision_matrix(gmrf)
    var_estimate = zeros(size(Q, 1))

    samples = [zeros(size(Q, 1)) for _ = 1:strategy.n_samples]
    for i = 1:strategy.n_samples
        rand!(strategy.rng, gmrf, samples[i])
        # Remove mean to get centered samples  
        samples[i] .-= mean(gmrf)
    end
    sample_mat = hcat(samples...)

    subsets = _build_disjoint_subsets(Q)

    for subset in subsets
        enclosure_idcs = _build_enclosure_idcs(Q, subset, strategy.enclosure_size)
        enclosure_p = symamd(Q[enclosure_idcs, enclosure_idcs])
        enclosure_idcs = enclosure_idcs[enclosure_p]

        interior_p = symamd(Q[subset, subset])
        interior = subset[interior_p]

        block_idcs = interior ∪ enclosure_idcs
        Q_block = Q[block_idcs, block_idcs]
        Q_block_row = Q[block_idcs, :]
        block_chol = cholesky(Symmetric(Q_block))

        var_estimate[interior] .=
            SelectedInversion.selinv_diag(block_chol)[1:length(interior)]

        κs = block_chol \ (Q_block_row * sample_mat - Q_block * sample_mat[block_idcs, :])
        var_estimate[interior] .+= var(κs, dims = 2)[1:length(interior), 1]
    end
    return var_estimate
end
