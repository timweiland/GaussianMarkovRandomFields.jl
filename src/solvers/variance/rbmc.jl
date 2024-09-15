using Random

export RBMCStrategy, compute_variance

struct RBMCStrategy <: AbstractVarianceStrategy
    n_samples::Int
    rng::Random.AbstractRNG

    function RBMCStrategy(n_samples::Int, rng::Random.AbstractRNG = Random.default_rng())
        new(n_samples, rng)
    end
end

function compute_variance(s::RBMCStrategy, solver::AbstractSolver)
    D = diag(to_matrix(gmrf(solver).precision))
    D⁻¹ = 1 ./ D
    var_estimate = zeros(length(D))
    cur_sample = zeros(length(D))
    Nₛ = s.n_samples
    Q = gmrf(solver).precision
    for i = 1:Nₛ
        compute_rand!(solver, s.rng, cur_sample)
        var_estimate .+= (1 / Nₛ) * (D⁻¹ .* (Q * cur_sample - D .* cur_sample)) .^ 2
    end
    return D⁻¹ + var_estimate
end
