using SparseArrays, LinearAlgebra, Distributions

struct CholeskySolver{G<:AbstractGMRF} <: AbstractSolver
    gmrf::G
    precision_chol::Union{Cholesky,SparseArrays.CHOLMOD.Factor}

    function CholeskySolver(gmrf::G) where {G}
        precision_chol = cholesky(to_matrix(precision_map(gmrf)))
        new{G}(gmrf, precision_chol)
    end
end

function compute_mean(s::CholeskySolver)
    return s.gmrf.mean
end

function compute_mean(s::CholeskySolver{<:LinearConditionalGMRF})
    x = s.gmrf
    residual = x.y - (x.A * x.prior.mean + x.b)
    return mean(x.prior) + s.precision_chol \ (x.A' * (x.Q_ϵ * residual))
end

function compute_variance(s::CholeskySolver)
    # Use sparse partial inverse (Takahashi recursions)
    return diag(sparseinv(s.precision_chol, depermute = true)[1])
end

function compute_rand!(s::CholeskySolver, rng::Random.AbstractRNG, x::AbstractVector)
    randn!(rng, x)
    x .= s.precision_chol.UP \ x
    x .+= mean(s.gmrf)
    return x
end

struct CholeskySolverBlueprint <: AbstractSolverBlueprint
    function CholeskySolverBlueprint()
        new()
    end
end

function construct_solver(::CholeskySolverBlueprint, gmrf::AbstractGMRF)
    return CholeskySolver(gmrf)
end
