using Random, AMD

export RBMCStrategy, BlockRBMCStrategy, compute_variance

struct RBMCStrategy <: AbstractVarianceStrategy
    n_samples::Int
    rng::Random.AbstractRNG

    function RBMCStrategy(n_samples::Int, rng::Random.AbstractRNG = Random.default_rng())
        new(n_samples, rng)
    end
end

function compute_variance(s::RBMCStrategy, solver::AbstractSolver)
    Q = to_matrix(precision_map(gmrf(solver)))
    D = Array(diag(Q))
    D⁻¹ = 1 ./ D

    samples = [zeros(Base.size(Q, 1)) for _ = 1:s.n_samples]
    for i = 1:s.n_samples
        compute_rand!(solver, s.rng, samples[i])
    end
    sample_mat = hcat(samples...)

    # Q̃ = Q - Diagonal(D)
    transformed_samples = D⁻¹ .* (Q * sample_mat - D .* sample_mat)
    return D⁻¹ + reshape(var(transformed_samples, dims = 2), length(D))
end

all_neighbors(Q, i) = findnz(Q[i, :])[1]
enclosure(Q, idcs) = vcat([all_neighbors(Q, i) for i in idcs]...)
function build_enclosure_idcs(Q, interior, enclosure_size = 1)
    enclosure_idcs = Int64[]
    new_idcs = copy(interior)
    explored = Set(interior)
    for _ = 1:enclosure_size
        new_idcs = Set(enclosure(Q, new_idcs))
        new_idcs = setdiff(new_idcs, explored)
        append!(enclosure_idcs, collect(new_idcs))
        explored = explored ∪ new_idcs
    end
    return enclosure_idcs
end

function build_disjoint_subsets(Q)
    visited = zeros(Bool, Base.size(Q, 1))
    subsets = []
    for i = 1:Base.size(Q, 1)
        if !visited[i]
            idcs = findnz(Q[i, :])[1]
            visited[idcs] .= true
            push!(subsets, idcs)
        end
    end
    return subsets
end

struct BlockRBMCStrategy <: AbstractVarianceStrategy
    n_samples::Int
    rng::Random.AbstractRNG
    enclosure_size::Int

    function BlockRBMCStrategy(
        n_samples::Int,
        rng::Random.AbstractRNG = Random.default_rng(),
        enclosure_size::Int = 1,
    )
        new(n_samples, rng, enclosure_size)
    end
end

function compute_variance(s::BlockRBMCStrategy, solver::AbstractSolver)
    Q = to_matrix(gmrf(solver).precision)

    var_estimate = zeros(Base.size(Q, 1))

    samples = [zeros(Base.size(Q, 1)) for _ = 1:s.n_samples]
    for i = 1:s.n_samples
        compute_rand!(solver, s.rng, samples[i])
    end
    sample_mat = hcat(samples...)

    subsets = build_disjoint_subsets(Q)

    for subset in subsets
        enclosure_idcs = build_enclosure_idcs(Q, subset, s.enclosure_size)
        enclosure_p = symamd(Q[enclosure_idcs, enclosure_idcs])
        enclosure_idcs = enclosure_idcs[enclosure_p]

        interior_p = symamd(Q[subset, subset])
        interior = subset[interior_p]

        block_idcs = interior ∪ enclosure_idcs
        Q_block = Q[block_idcs, block_idcs]
        Q_block_row = Q[block_idcs, :]
        block_chol = cholesky(Symmetric(Q_block))

        var_estimate[interior] .=
            diag(sparseinv(block_chol, depermute = true)[1])[1:length(interior)]

        κs = block_chol \ (Q_block_row * sample_mat - Q_block * sample_mat[block_idcs, :])
        var_estimate[interior] .+= var(κs, dims = 2)[1:length(interior), 1]
    end
    return var_estimate
end
