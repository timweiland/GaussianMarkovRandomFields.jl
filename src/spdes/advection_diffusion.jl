using Ferrite, LinearAlgebra, SparseArrays

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

    function AdvectionDiffusionSPDE{D}(
        κ::Real,
        α::Rational,
        H::AbstractMatrix,
        γ::AbstractVector,
        c::Real,
        τ::Real,
    ) where {D}
        κ > 0 || throw(ArgumentError("κ must be positive"))
        α >= 0 || throw(ArgumentError("α must be non-negative"))
        τ > 0 || throw(ArgumentError("τ must be positive"))
        (D >= 1 && isinteger(D)) || throw(ArgumentError("D must be a positive integer"))
        new{D}(κ, α, H, γ, c, τ)
    end
end

function assemble_M_G_B_matrices(
    cellvalues::CellScalarValues,
    dh::DofHandler,
    interpolation,
    H,
    γ;
    streamline_diffusion = false,
    h = 0.1,
)
    M, G, B, S = create_sparsity_pattern(dh),
    create_sparsity_pattern(dh),
    create_sparsity_pattern(dh),
    create_sparsity_pattern(dh)

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
        Ge = assemble_diffusion_matrix(Ge, cellvalues; diffusion_matrix = H)
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
    colored_noise = false,
    streamline_diffusion = false,
    h = 0.1,
) where {D}
    cellvalues =
        CellScalarValues(discretization.quadrature_rule, discretization.interpolation)
    if streamline_diffusion
        M, G, B, S = assemble_M_G_B_matrices(
            cellvalues,
            discretization.dof_handler,
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
            discretization.interpolation,
            spde.H,
            spde.γ,
        )
        τ = spde.τ
    end
    M⁻¹ = spdiagm(0 => 1 ./ diag(M))
    K = (spde.κ^2 * M + G)^spde.α

    matern_spde = MaternSPDE{D}(5.0, 1)
    x₀ = discretize(matern_spde, discretization)
    Q_s = x₀.precision

    if streamline_diffusion
        E_fn = dt -> M + (dt / spde.c) * (K + B + S)
    else
        E_fn = dt -> M + (dt / spde.c) * (K + B)
    end

    β = dt -> (spde.c / (dt * τ^2))
    if !colored_noise
        F⁻¹_fn = dt -> β(dt) * E_fn(dt)' * M⁻¹ * E_fn(dt)
        AF⁻¹A_fn = dt -> β(dt) * M
        F⁻¹A_fn = dt -> β(dt) * E_fn(dt)'
    else
        M⁻¹Q = M⁻¹ * Q_s
        M⁻¹QM⁻¹ = M⁻¹Q * M⁻¹
        F⁻¹_fn = dt -> β(dt) * E_fn(dt)' * M⁻¹QM⁻¹ * E_fn(dt)
        AF⁻¹A_fn = dt -> β(dt) * Q_s
        F⁻¹A_fn = dt -> β(dt) * E_fn(dt)' * M⁻¹Q
    end

    X = joint_ssm(x₀, AF⁻¹A_fn, F⁻¹_fn, F⁻¹A_fn, ts)
    X = ConstantMeshSTGMRF(X.mean, X.precision, discretization)
    return X
end
