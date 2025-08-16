using Ferrite, LinearAlgebra, SparseArrays, LinearMaps

export AdvectionDiffusionSPDE, discretize

@doc raw"""
    AdvectionDiffusionSPDE{D}(κ::Real, α::Rational, H::AbstractMatrix,
    γ::AbstractVector, c::Real, τ::Real) where {D}

Spatiotemporal advection-diffusion SPDE as proposed in [Clarotto2024](@cite):

```math
\left[ \frac{∂}{∂t} + \frac{1}{c} \left( κ^2 - ∇ ⋅ H ∇ \right)^\alpha
+ \frac{1}{c} γ ⋅ ∇ \right] X(t, s) = \frac{τ}{\sqrt{c}} Z(t, s),
```

where Z(t, s) is spatiotemporal noise which may be colored.
"""
struct AdvectionDiffusionSPDE{D} <: SPDE
    κ::Real
    α::Rational
    H::AbstractMatrix
    γ::AbstractVector
    c::Real
    τ::Real
    spatial_spde::SPDE
    initial_spde::SPDE

    function AdvectionDiffusionSPDE{D}(;
            κ::Real = 1.0,
            α::Rational = 1 // 1,
            H::AbstractMatrix = sparse(I, (D, D)),
            γ::AbstractVector,
            c::Real = 1.0,
            τ::Real = 1.0,
            spatial_spde = MaternSPDE{D}(κ = κ, smoothness = 1, diffusion_factor = H),
            initial_spde = MaternSPDE{D}(κ = κ, smoothness = 2, diffusion_factor = H),
        ) where {D}
        κ >= 0 || throw(ArgumentError("κ must be non-negative"))
        α >= 0 || throw(ArgumentError("α must be non-negative"))
        τ > 0 || throw(ArgumentError("τ must be positive"))
        # νₛ > 0 || throw(ArgumentError("νₛ must be positive"))
        (D >= 1 && isinteger(D)) || throw(ArgumentError("D must be a positive integer"))
        return new{D}(κ, α, H, γ, c, τ, spatial_spde, initial_spde)
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
        Ferrite.reinit!(cellvalues, cell)
        Me = assemble_mass_matrix(Me, cellvalues, interpolation; lumping = true)
        Ge = assemble_diffusion_matrix(Ge, cellvalues; diffusion_factor = H)
        Be = assemble_advection_matrix(Be, cellvalues; advection_velocity = γ)
        if streamline_diffusion
            cell_volume = 0.0
            for qp in 1:getnquadpoints(cellvalues)
                dΩ = getdetJdV(cellvalues, qp)
                cell_volume += dΩ
            end
            Se = assemble_streamline_diffusion_matrix(Se, cellvalues, γ, cell_volume)
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
        mean_offset = 0.0,
        prescribed_noise = 1.0e-4,
        algorithm = nothing,
    ) where {D}
    if norm(spde.γ) ≈ 0.0
        # SD changes nothing for zero advection
        streamline_diffusion = false
    end

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
        )
    else
        M, G, B = assemble_M_G_B_matrices(
            cellvalues,
            discretization.dof_handler,
            ch,
            discretization.interpolation,
            spde.H,
            spde.γ,
        )
    end
    τ = spde.τ
    apply!(M, zeros(Base.size(M, 2)), ch)
    apply!(G, zeros(Base.size(M, 2)), ch)
    K = (spde.κ^2 * M + G)^spde.α

    xₛ = discretize(spde.spatial_spde, discretization)
    x₀ = discretize(spde.initial_spde, discretization)

    apply!(B, zeros(Base.size(B, 2)), ch)
    propagation_mat = K + B
    if streamline_diffusion
        apply!(S, zeros(Base.size(S, 2)), ch)
        propagation_mat += S
    end
    G_fn = dt -> LinearMap(M + (dt / spde.c) * propagation_mat)
    M⁻¹ = spdiagm(0 => 1 ./ diag(M))

    noise_mat = spdiagm(0 => fill(τ / sqrt(spde.c), Base.size(M, 2)))

    Nₛ = Base.size(propagation_mat, 2)
    total_ndofs = Nₛ * length(ts)
    mean_offset = fill(mean_offset, total_ndofs)

    inv_noise_mat = spdiagm(0 => 1 ./ diag(noise_mat))
    β = dt -> sqrt(dt) * noise_mat
    β⁻¹ = dt -> (1 / sqrt(dt)) * inv_noise_mat

    ssm = ImplicitEulerSSM(
        x₀,
        G_fn,
        dt -> LinearMap(M),
        dt -> LinearMap(M⁻¹),
        β,
        β⁻¹,
        xₛ,
        ts,
        discretization.constraint_handler,
        discretization.constraint_noise,
    )
    X = joint_ssm(ssm)
    X = X + mean_offset
    X = ImplicitEulerConstantMeshSTGMRF(
        X,
        discretization,
        ssm
    )
    return X
end
