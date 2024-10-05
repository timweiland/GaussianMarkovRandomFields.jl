using LinearAlgebra, Preconditioners

export GaussNewtonOptimizer,
    GNLinearSolverBlueprint, GNCGSolverBlueprint, GNCholeskySolverBlueprint

abstract type GNLinearSolverBlueprint end

struct GNCholeskySolverBlueprint <: GNLinearSolverBlueprint
    perm::Union{Nothing,Vector{Int}}

    function GNCholeskySolverBlueprint(perm::Union{Nothing,Vector{Int}} = nothing)
        new(perm)
    end
end

struct GNCGSolverBlueprint <: GNLinearSolverBlueprint
    maxiter::Int
    reltol::Real
    abstol::Real
    preconditioner_fn::Function

    function GNCGSolverBlueprint(
        maxiter::Int = 100,
        reltol::Real = 1e-6,
        abstol::Real = 1e-6,
        preconditioner_fn::Function = A -> Preconditioners.Identity(),
    )
        new(maxiter, reltol, abstol, preconditioner_fn)
    end
end

mutable struct GaussNewtonOptimizer
    μ_prior::AbstractVector
    Q_prior::LinearMap
    f::Function
    J_fn::Function
    noise::Real
    y::AbstractVector
    xₖ::AbstractVector
    rₖ::AbstractVector # Observation residual
    r_norm_history::Vector{Real}
    obj_val_history::Vector{Real}
    Jₖ::AbstractMatrix
    solver_bp::GNLinearSolverBlueprint
    Qx_prior::AbstractVector
    Q_mat::AbstractMatrix

    function GaussNewtonOptimizer(
        μ_prior::AbstractVector,
        Q_prior::LinearMap,
        f::Function,
        J_fn::Function,
        noise::Real,
        y::AbstractVector,
        x₀::AbstractVector = μ_prior,
        solver_bp::GNLinearSolverBlueprint = GNCholeskySolverBlueprint(),
    )
        J₀ = J_fn(x₀)
        r₀ = y - f(x₀)

        prior_diff = x₀ - μ_prior
        obj_val = dot(prior_diff, Q_prior * prior_diff) + dot(r₀, noise * r₀)
        new(
            μ_prior,
            Q_prior,
            f,
            J_fn,
            noise,
            y,
            x₀,
            r₀,
            [norm(r₀)],
            [obj_val],
            J₀,
            solver_bp,
            Q_prior * μ_prior,
            to_matrix(Q_prior),
        )
    end
end

function step(optim::GaussNewtonOptimizer)
    _step(optim, optim.solver_bp)
end

function _update(optim::GaussNewtonOptimizer)
    optim.Jₖ = optim.J_fn(optim.xₖ)
    optim.rₖ = optim.y - optim.f(optim.xₖ)
    push!(optim.r_norm_history, norm(optim.rₖ))
    prior_diff = Array(optim.xₖ - optim.μ_prior)
    obj_val =
        dot(prior_diff, optim.Q_prior * prior_diff) + dot(optim.rₖ, optim.noise * optim.rₖ)
    push!(optim.obj_val_history, obj_val)
end

function _step(optim::GaussNewtonOptimizer, solver_bp::GNCholeskySolverBlueprint)
    A = Symmetric(optim.Q_mat + optim.noise * optim.Jₖ' * optim.Jₖ)
    rhs = optim.Qx_prior + optim.noise * optim.Jₖ' * Array(optim.Jₖ * optim.xₖ + optim.rₖ)
    A_chol = cholesky(A; perm = solver_bp.perm)

    optim.xₖ = A_chol \ rhs
    _update(optim)
end

function _step(optim::GaussNewtonOptimizer, solver_bp::GNCGSolverBlueprint)
    A = Symmetric(optim.Q_mat + optim.noise * optim.Jₖ' * optim.Jₖ)

    rhs = optim.Qx_prior + optim.noise * optim.Jₖ' * Array(optim.Jₖ * optim.xₖ + optim.rₖ)

    Pl = solver_bp.preconditioner_fn(sparse(A))
    cg!(
        optim.xₖ,
        A,
        rhs;
        maxiter = solver_bp.maxiter,
        reltol = solver_bp.reltol,
        abstol = solver_bp.abstol,
        verbose = true,
        Pl = Pl,
    )
    _update(optim)
end
