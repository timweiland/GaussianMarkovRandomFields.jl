using SparseArrays, LinearAlgebra, Distributions, IterativeSolvers

export CGSolver, CGSolverBlueprint, construct_solver, compute_mean

struct CGSolver{G<:AbstractGMRF} <: AbstractSolver
    gmrf::G
    reltol::Real
    abstol::Real
    maxiter::Int
    Pl::Union{Identity, AbstractPreconditioner}

    function CGSolver(gmrf::G, reltol::Real, abstol::Real, maxiter::Int, Pl::Union{Identity, AbstractPreconditioner}) where {G}
        new{G}(gmrf, reltol, abstol, maxiter, Pl)
    end
end

function compute_mean(s::CGSolver)
    return s.gmrf.mean
end

function compute_mean(s::CGSolver{<:LinearConditionalGMRF})
    x = s.gmrf
    residual = x.y - (x.A * x.prior.mean + x.b)
    rhs = x.A' * (x.Q_ϵ * residual)
    return mean(x.prior) + cg(
        x.precision,
        Array(rhs);
        maxiter = s.maxiter,
        reltol = s.reltol,
        abstol = s.abstol,
        verbose = true,
        Pl=s.Pl,
    )
end

function compute_rand!(
    s::CGSolver{<:LinearConditionalGMRF},
    rng::Random.AbstractRNG,
    x::AbstractVector,
)
    rand!(rng, s.gmrf.prior, x)
    x_cond = s.gmrf
    prior_mean = mean(x_cond.prior)
    residual = x_cond.A * (x - prior_mean) - x_cond.b
    rhs = x_cond.A' * (x_cond.Q_ϵ * residual)
    sample_kriging_mean =
        prior_mean + cg(
            s.gmrf.precision,
            Array(rhs);
            maxiter = s.maxiter,
            reltol = s.reltol,
            abstol = s.abstol,
            verbose = true,
            Pl=s.Pl,
        )

    sample_resid = sample_kriging_mean - x
    x .= mean(x_cond) + sample_resid
    return x
end

function compute_rand!(
    s::CGSolver{<:ConstantMeshSTGMRF},
    rng::Random.AbstractRNG,
    x::AbstractVector,
)
    ssm = s.gmrf.ssm
    xₖ = zeros(size(ssm.x₀))
    rand!(rng, ssm.x₀, xₖ) # Sample from x₀
    ts = ssm.ts
    t_prev = ts[1]
    z = zeros(size(xₖ))
    xs = [xₖ]
    for t in ts[2:end]
        Δt = t - t_prev
        G = ssm.G(Δt)
        M = ssm.M(Δt)
        β = ssm.β(Δt)
        rand!(rng, ssm.spatial_noise, z)
        rhs = M * (xₖ + β * z)
        xₖ = cg(
            G,
            Array(rhs);
            maxiter = s.maxiter,
            reltol = s.reltol,
            abstol = s.abstol,
            verbose = true,
            Pl=s.Pl,
        )
        push!(xs, xₖ)
        t_prev = t
    end
    x .= vcat(xs...)
    return x
end

function default_preconditioner_strategy(::AbstractGMRF)
    return Identity()
end

function default_preconditioner_strategy(x::LinearConditionalGMRF{<:ConstantMeshSTGMRF})
    prior_diag_blocks = x.prior.precision.diagonal_blocks
    update_term = sparse(x.A' * (x.Q_ϵ * x.A))
    start = 1
    stop = 0
    sparse_blocks = []
    for block in prior_diag_blocks
        stop += size(block, 1)
        push!(sparse_blocks, sparse(block) + update_term[start:stop, start:stop])
        start = stop + 1
    end
    Ps = [FullCholeskyPreconditioner(b) for b in sparse_blocks]
    return BlockJacobiPreconditioner(Ps)
end

struct CGSolverBlueprint <: AbstractSolverBlueprint
    reltol::Real
    abstol::Real
    maxiter::Int
    preconditioner_strategy::Function

    function CGSolverBlueprint(
        reltol::Real = sqrt(eps(Float64)),
        abstol::Real = 0.0,
        maxiter::Int = 1000,
        preconditioner_strategy::Function = default_preconditioner_strategy,
    )
        new(reltol, abstol, maxiter, preconditioner_strategy)
    end
end

function construct_solver(bp::CGSolverBlueprint, gmrf::AbstractGMRF)
    Pl = bp.preconditioner_strategy(gmrf)
    return CGSolver(gmrf, bp.reltol, bp.abstol, bp.maxiter, Pl)
end
