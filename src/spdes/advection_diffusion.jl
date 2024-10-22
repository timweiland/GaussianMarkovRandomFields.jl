using Ferrite, LinearAlgebra, SparseArrays, LinearMaps

export AdvectionDiffusionSPDE, discretize

@doc raw"""
    AdvectionDiffusionSPDE{D}(κ::Real, α::Rational, H::AbstractMatrix,
    γ::AbstractVector, c::Real, τ::Real) where {D}

Spatiotemporal advection-diffusion SPDE as proposed in [1]:

```math
\left[ \frac{∂}{∂t} + \frac{1}{c} \left( κ^2 - ∇ ⋅ H ∇ \right)^\alpha
+ \frac{1}{c} γ ⋅ ∇ \right] X(t, s) = \frac{τ}{\sqrt{c}} Z(t, s),
```

where Z(t, s) is spatiotemporal noise which may be colored.

[1] Clarotto, Lucia, et al. "The SPDE approach for spatio-temporal datasets with
advection and diffusion." Spatial Statistics (2024): 100847.
"""
struct AdvectionDiffusionSPDE{D} <: SPDE
    κ::Real
    α::Rational
    H::AbstractMatrix
    γ::AbstractVector
    c::Real
    τ::Real
    νₛ::Real
    κₛ::Real

    function AdvectionDiffusionSPDE{D}(
        κ::Real,
        α::Rational,
        H::AbstractMatrix,
        γ::AbstractVector,
        c::Real,
        τ::Real,
        νₛ::Real,
        κₛ::Real = κ,
    ) where {D}
        κ >= 0 || throw(ArgumentError("κ must be non-negative"))
        κₛ > 0 || throw(ArgumentError("κ must be positive"))
        α >= 0 || throw(ArgumentError("α must be non-negative"))
        τ > 0 || throw(ArgumentError("τ must be positive"))
        νₛ > 0 || throw(ArgumentError("νₛ must be positive"))
        (D >= 1 && isinteger(D)) || throw(ArgumentError("D must be a positive integer"))
        new{D}(κ, α, H, γ, c, τ, νₛ, κₛ)
    end
end

function assemble_M_G_B_matrices(
    cellvalues::CellValues,
    dh::DofHandler,
    ch,
    interpolation,
    H,
    γ;
    streamline_diffusion = false,
    h = 0.1,
)
    M, G, B, S = allocate_matrix(dh, ch),
    allocate_matrix(dh, ch),
    allocate_matrix(dh, ch),
    allocate_matrix(dh, ch)

    n_basefuncs = getnbasefunctions(cellvalues)
    Me = spzeros(n_basefuncs, n_basefuncs)
    Ge = spzeros(n_basefuncs, n_basefuncs)
    Be = spzeros(n_basefuncs, n_basefuncs)
    if streamline_diffusion
        Se = spzeros(n_basefuncs, n_basefuncs)
    end

    M_assembler = start_assemble(M)
    G_assembler = start_assemble(G)
    B_assembler = start_assemble(B)
    if streamline_diffusion
        S_assembler = start_assemble(S)
    end

    for cell in CellIterator(dh)
        reinit!(cellvalues, cell)
        Me = assemble_mass_matrix(Me, cellvalues, interpolation; lumping = true)
        Ge = assemble_diffusion_matrix(Ge, cellvalues; diffusion_factor = H)
        Be = assemble_advection_matrix(Be, cellvalues; advection_velocity = γ)
        if streamline_diffusion
            Se = assemble_streamline_diffusion_matrix(Se, cellvalues, γ, h)
            assemble!(S_assembler, celldofs(cell), Se)
        end
        assemble!(M_assembler, celldofs(cell), Me)
        assemble!(G_assembler, celldofs(cell), Ge)
        assemble!(B_assembler, celldofs(cell), Be)
    end
    if streamline_diffusion
        return M, G, B, S
    end
    return M, G, B
end

"""
    discretize(spde::AdvectionDiffusionSPDE, discretization::FEMDiscretization,
    ts::AbstractVector{Float64}; colored_noise = false,
    streamline_diffusion = false, h = 0.1) where {D}

Discretize an advection-diffusion SPDE using a constant spatial mesh.
Streamline diffusion is an optional stabilization scheme for advection-dominated
problems, which are known to be unstable.
When using streamline diffusion, `h` may be passed to specify the
mesh element size.
"""
function discretize(
    spde::AdvectionDiffusionSPDE{D},
    discretization::FEMDiscretization{D},
    ts::AbstractVector{Float64};
    streamline_diffusion = false,
    h = 0.1,
    mean_offset = 0.0,
    prescribed_noise = 1e-4,
    solver_bp::AbstractSolverBlueprint = DefaultSolverBlueprint(),
) where {D}
    cellvalues = CellValues(
        discretization.quadrature_rule,
        discretization.interpolation,
        discretization.geom_interpolation,
    )
    ch = discretization.constraint_handler
    if streamline_diffusion
        M, G, B, S = assemble_M_G_B_matrices(
            cellvalues,
            discretization.dof_handler,
            ch,
            discretization.interpolation,
            spde.H,
            spde.γ;
            streamline_diffusion = true,
            h = h,
        )
        τ =
            spde.τ *
            norm(spde.H + h * (1 / norm(spde.γ)) * (spde.γ .* spde.γ'))^(1 / 4) *
            norm(spde.H)^(1 / 4)
    else
        M, G, B = assemble_M_G_B_matrices(
            cellvalues,
            discretization.dof_handler,
            ch,
            discretization.interpolation,
            spde.H,
            spde.γ,
        )
        τ = spde.τ
    end
    apply!(M, zeros(Base.size(M, 2)), ch)
    apply!(G, zeros(Base.size(M, 2)), ch)
    K = (spde.κ^2 * M + G)^spde.α

    matern_spde_spatial = MaternSPDE{D}(spde.κₛ, spde.νₛ, 1.0, spde.H)
    xₛ = discretize(matern_spde_spatial, discretization)

    matern_spde_t₀ =
        MaternSPDE{D}(spde.κₛ, spde.α + α(matern_spde_spatial) - D // 2, 1.0, spde.H)
    x₀ = discretize(matern_spde_t₀, discretization)

    apply!(B, zeros(Base.size(B, 2)), ch)
    propagation_mat = K + B
    if streamline_diffusion
        apply!(S, zeros(Base.size(S, 2)), ch)
        propagation_mat += S
    end
    for dof in ch.prescribed_dofs
        M[dof, dof] = prescribed_noise # TODO
        propagation_mat[dof, dof] = 0.0
    end
    G_fn = dt -> LinearMap(M + (dt / spde.c) * propagation_mat)
    M⁻¹ = spdiagm(0 => 1 ./ diag(M))

    noise_mat = spdiagm(0 => fill(τ / sqrt(spde.c), Base.size(M, 2)))

    Nₛ = Base.size(propagation_mat, 2)
    total_ndofs = Nₛ * length(ts)
    mean_offset = fill(mean_offset, total_ndofs)
    for dof in ch.prescribed_dofs
        noise_mat[dof, dof] = prescribed_noise
        st_dofs = dof:Nₛ:total_ndofs
        mean_offset[st_dofs] .= 0.0
    end
    inv_noise_mat = spdiagm(0 => 1 ./ diag(noise_mat))
    β = dt -> sqrt(dt) * noise_mat
    β⁻¹ = dt -> (1 / sqrt(dt)) * inv_noise_mat
    # TODO
    # β⁻¹ = dt -> sqrt(spde.c / (dt * τ^2))
    # β = dt -> sqrt((dt * τ^2) / spde.c)

    ssm = ImplicitEulerSSM(
        (x₀ isa ConstrainedGMRF) ? x₀.inner_gmrf : x₀,
        G_fn,
        dt -> LinearMap(M),
        dt -> LinearMap(M⁻¹),
        β,
        β⁻¹,
        xₛ,
        ts,
    )
    X = joint_ssm(ssm)
    X = ImplicitEulerConstantMeshSTGMRF(
        X.mean .+ mean_offset,
        X.precision,
        discretization,
        ssm,
        solver_bp,
    )
    if length(ch.prescribed_dofs) > 0
        return ConstrainedGMRF(X, ch)
    end
    return X
end
