using LinearAlgebra, Preconditioners

export GaussNewtonOptimizer, optimize

mutable struct GaussNewtonOptimizer
    μ_prior::AbstractVector
    Q_prior::LinearMap
    f_and_Jf::Function
    noise::Real
    y::AbstractVector
    xₖ::AbstractVector
    r_obsₖ::AbstractVector # Observation residual
    r_priorₖ::AbstractVector # Prior residual
    ∇objₖ::AbstractVector # Gradient of the objective function
    r_obs_norm_history::Vector{Real}
    obj_val_history::Vector{Real}
    ∇obj_val_norm_history::Vector{Real}
    newton_decrement::Real
    Jₖ::AbstractMatrix
    solver_bp::GNLinearSolverBlueprint
    Qx_prior::AbstractVector
    Q_mat::AbstractMatrix
    line_search::AbstractLineSearch
    stopping_criterion::AbstractStoppingCriterion

    function GaussNewtonOptimizer(
        μ_prior::AbstractVector,
        Q_prior::LinearMap,
        f_and_Jf::Function,
        noise::Real,
        y::AbstractVector,
        x₀::AbstractVector = μ_prior;
        solver_bp::GNLinearSolverBlueprint = GNCholeskySolverBlueprint(),
        line_search::AbstractLineSearch = BacktrackingLineSearch(),
        stopping_criterion::AbstractStoppingCriterion = OrCriterion([
            NewtonDecrementCriterion(1e-5),
            StepNumberCriterion(15),
        ]),
    )
        f₀, J₀ = f_and_Jf(x₀)
        r_obs₀ = Array(f₀ - y)
        r_prior₀ = Array(x₀ - μ_prior)
        ∇obj₀ = Q_prior * r_prior₀ + noise * J₀' * r_obs₀

        obj_val = 0.5 * (dot(r_prior₀, Q_prior * r_prior₀) + dot(r_obs₀, noise * r_obs₀))
        new(
            μ_prior,
            Q_prior,
            f_and_Jf,
            noise,
            y,
            x₀,
            r_obs₀,
            r_prior₀,
            ∇obj₀,
            [norm(r_obs₀)],
            [obj_val],
            [norm(∇obj₀)],
            Inf,
            J₀,
            solver_bp,
            Q_prior * μ_prior,
            to_matrix(Q_prior),
            line_search,
            stopping_criterion,
        )
    end
end

function obj_fn(
    optim::GaussNewtonOptimizer,
    x::AbstractVector,
    r_prior::Union{Nothing,AbstractVector} = nothing,
    r_obs::Union{Nothing,AbstractVector} = nothing,
)
    if r_obs === nothing
        f, _ = optim.f_and_Jf(x) # TODO: A bit wasteful to compute J here
        r_obs = Array(f - optim.y)
    end
    if r_prior === nothing
        r_prior = Array(x - optim.μ_prior)
    end
    return 0.5 * (dot(r_prior, optim.Q_prior * r_prior) + dot(r_obs, optim.noise * r_obs))
end

function ∇obj_fn(
    optim::GaussNewtonOptimizer,
    x::AbstractVector,
    J::Union{Nothing,AbstractMatrix} = nothing,
    r_prior::Union{Nothing,AbstractVector} = nothing,
    r_obs::Union{Nothing,AbstractVector} = nothing,
)
    if J === nothing || r_obs === nothing
        f, J = optim.f_and_Jf(x)
        r_obs = Array(f - optim.y)
    end
    if r_prior === nothing
        r_prior = Array(x - optim.μ_prior)
    end
    return optim.Q_prior * r_prior + optim.noise * J' * r_obs
end


function step(optim::GaussNewtonOptimizer)
    p = _compute_direction(optim, optim.solver_bp)
    optim.newton_decrement = -dot(optim.∇objₖ, p)
    optim.xₖ =
        line_search(x -> obj_fn(optim, x), optim.∇objₖ, optim.xₖ, p, optim.line_search)
    _update(optim)
end

function _update(optim::GaussNewtonOptimizer)
    fₖ, optim.Jₖ = optim.f_and_Jf(optim.xₖ)
    optim.r_obsₖ = Array(fₖ - optim.y)
    optim.r_priorₖ = Array(optim.xₖ - optim.μ_prior)
    optim.∇objₖ = ∇obj_fn(optim, optim.xₖ, optim.Jₖ, optim.r_priorₖ, optim.r_obsₖ)
    obj_val = obj_fn(optim, optim.xₖ, optim.r_priorₖ, optim.r_obsₖ)
    push!(optim.r_obs_norm_history, norm(optim.r_obsₖ))
    push!(optim.∇obj_val_norm_history, norm(optim.∇objₖ))
    push!(optim.obj_val_history, obj_val)
end

function _compute_direction(
    optim::GaussNewtonOptimizer,
    solver_bp::GNCholeskySolverBlueprint,
)
    H = Symmetric(optim.Q_mat + optim.noise * optim.Jₖ' * optim.Jₖ)
    H_chol = cholesky(H; perm = solver_bp.perm)

    d = -(H_chol \ optim.∇objₖ)

    return d
end

function _compute_direction(optim::GaussNewtonOptimizer, solver_bp::GNCGSolverBlueprint)
    H = Symmetric(optim.Q_mat + optim.noise * optim.Jₖ' * optim.Jₖ)
    Pl = solver_bp.preconditioner_fn(sparse(H))

    return cg(
        H,
        optim.∇objₖ;
        maxiter = solver_bp.maxiter,
        reltol = solver_bp.reltol,
        abstol = solver_bp.abstol,
        verbose = true,
        Pl = Pl,
    )
end

function optimize(optim::GaussNewtonOptimizer)
    while !_should_stop(optim, optim.stopping_criterion)
        step(optim)
    end
end

function _should_stop(optim::GaussNewtonOptimizer, criterion::AbstractStoppingCriterion)
    return optim.newton_decrement < criterion.threshold
end

function _should_stop(optim::GaussNewtonOptimizer, criterion::NewtonDecrementCriterion)
    return optim.newton_decrement < criterion.threshold
end

function _should_stop(optim::GaussNewtonOptimizer, criterion::StepNumberCriterion)
    return length(optim.obj_val_history) >= criterion.max_steps
end

function _should_stop(optim::GaussNewtonOptimizer, criterion::OrCriterion)
    return any(map(c -> _should_stop(optim, c), criterion.criteria))
end
