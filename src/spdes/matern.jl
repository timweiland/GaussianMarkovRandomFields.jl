using Ferrite
using LinearAlgebra
using SparseArrays
using SpecialFunctions

export MaternSPDE,
    α, ndim, discretize, assemble_C_G_matrices, product_matern, range_to_κ, smoothness_to_ν

################################################################################
#    Whittle-Matérn
################################################################################
@doc raw"""
    MaternSPDE{D}(κ::Real, ν::Union{Integer, Rational}) where D

The Whittle-Matérn SPDE is given by

```math
(κ^2 - Δ)^{\frac{α}{2}} u(x) = 𝒲(x), \quad \left( x \in \mathbb{R}^d,
α = ν + \frac{d}{2} \right),
```

where Δ is the Laplacian operator, $κ > 0$, $ν > 0$.

The stationary solutions to this SPDE are Matérn processes.
"""
struct MaternSPDE{D, Tv<:Real, Ti<:Integer} <: SPDE
    κ::Tv
    ν::Rational{Ti}
    σ²::Tv
    diffusion_factor::Matrix{Tv}
end

function MaternSPDE{D}(;
    κ::Union{Real, Nothing} = nothing,
    ν::Union{Integer, Rational, Nothing} = nothing,
    range::Union{Real, Nothing} = nothing,
    smoothness::Union{Integer, Nothing} = nothing,
    σ²::Real = 1.0,
    diffusion_factor::Union{AbstractMatrix, Nothing} = nothing,
) where {D}
    # Same logic as before, but D is fixed from the type
    ((κ === nothing) ⊻ (range === nothing)) || throw(ArgumentError("Either κ or range must be specified"))
    ((ν === nothing) ⊻ (smoothness === nothing)) || throw(ArgumentError("Either ν or smoothness must be specified"))

    if ν === nothing
        ν = smoothness_to_ν(smoothness, D)
    end
    if κ === nothing
        κ = range_to_κ(range, ν)
    end

    Tv = promote_type(typeof(κ), typeof(σ²))
    Ti = ν isa Rational ? typeof(denominator(ν)) : typeof(ν)
    ν_r = ν isa Rational ? ν : Rational{Ti}(ν)

    κ > 0 || throw(ArgumentError("κ must be positive"))
    ν_r >= 0 || throw(ArgumentError("ν must be non-negative"))
    σ² > 0 || throw(ArgumentError("σ² must be positive"))

    diffusion_factor_mat = diffusion_factor === nothing ? Matrix{Tv}(I, D, D) : Matrix{Tv}(diffusion_factor)

    return MaternSPDE{D, Tv, Ti}(κ, ν_r, σ², diffusion_factor_mat)
end


α(𝒟::MaternSPDE{D}) where {D} = 𝒟.ν + D // 2
ndim(::MaternSPDE{D}) where {D} = D

function assemble_C_G_matrices(
    cellvalues::CellValues,
    dh::DofHandler,
    interpolation,
    diffusion_factor::Matrix{Tv},
) where {Tv}
    C, G = allocate_matrix(SparseMatrixCSC{Tv, Int}, dh), allocate_matrix(SparseMatrixCSC{Tv, Int}, dh)

    n_basefuncs = getnbasefunctions(cellvalues)
    Ce = spzeros(Tv, n_basefuncs, n_basefuncs)
    Ge = spzeros(Tv, n_basefuncs, n_basefuncs)

    C_assembler = start_assemble(C)
    G_assembler = start_assemble(G)

    for cell in CellIterator(dh)
        reinit!(cellvalues, cell)
        Ce = assemble_mass_matrix(Ce, cellvalues, interpolation; lumping = false)
        Ge = assemble_diffusion_matrix(Ge, cellvalues; diffusion_factor = diffusion_factor)
        assemble!(C_assembler, celldofs(cell), Ce)
        assemble!(G_assembler, celldofs(cell), Ge)
    end
    C = lump_matrix(C, interpolation)

    return C, G
end

function _inner_cholesky(A::LinearMap, ::AbstractSolverBlueprint)
    return linmap_cholesky(Val{:default}(), A)
end

function _inner_cholesky(A::LinearMap, ::CholeskySolverBlueprint{:autodiffable})
    return linmap_cholesky(Val{:autodiffable}(), A)
end

"""
    matern_precision(C_inv::AbstractMatrix, K::AbstractMatrix, α::Integer)

Compute the precision matrix of a GMRF discretization of a Matérn SPDE.
Implements the recursion described in [Lindgren2011](@cite).

# Arguments
- `C_inv::AbstractMatrix`: The inverse of the (possibly lumped) mass matrix.
- `K::AbstractMatrix`: The stiffness matrix.
- `α::Integer`: The parameter α = ν + d/2 of the Matérn SPDE.
"""
function matern_mean_precision(
    C::SparseMatrixCSC{Tv, Ti},
    K::SparseMatrixCSC{Tv, Ti},
    α::Integer,
    ch,
    constraint_noise,
    solver_bp::AbstractSolverBlueprint,
    scaling_factor::Real = 1.0,
) where {Tv, Ti}
    if α < 1
        throw(ArgumentError("α must be positive and non-zero"))
    end
    C_inv = spdiagm(0 => 1 ./ diag(C))
    scale_mat = scaling_factor * sparse(Tv, 1.0*I, size(K)...)
    scale_mat_sqrt = sqrt(scaling_factor) * sparse(Tv, I, size(K)...)
    for dof in ch.prescribed_dofs
        scale_mat[dof, dof] = 1.0
        scale_mat_sqrt[dof, dof] = 1.0
    end

    if α == 1
        f_rhs = zeros(Tv, size(C, 1))

        for dof in ch.prescribed_dofs
            constraint_idx = ch.dofmapping[dof]
            if ch.dofcoefficients[constraint_idx] !== nothing
                # Not supported, throw error
                throw(ArgumentError("Non-Dirichlet BCs not supported for odd α"))
            end
            inhomogeneity = ch.inhomogeneities[constraint_idx]
            if inhomogeneity !== nothing
                f_rhs[dof] = inhomogeneity
            end
            K[dof, :] .= 0.0
            K[:, dof] .= 0.0
            K[dof, dof] = 1.0
        end
        μ = K \ f_rhs

        for dof in ch.prescribed_dofs
            constraint_idx = ch.dofmapping[dof]
            K[dof, dof] = constraint_noise[constraint_idx]^(-2)
        end

        Q_sym = LinearMap(Symmetric(scale_mat * K))
        Q_cho_sqrt = CholeskySqrt(_inner_cholesky(Q_sym, solver_bp))
        Q_alpha_1 = LinearMapWithSqrt(Q_sym, Q_cho_sqrt)
        return μ, Q_alpha_1
    elseif α == 2
        C_inv_sqrt = spdiagm(0 => sqrt.(diag(C_inv)))
        f_rhs = zeros(Tv, size(C, 1))
        Q_rhs = C_inv
        Q_rhs_sqrt = C_inv_sqrt
    else
        f_inner, Q_inner =
            matern_mean_precision(copy(C), copy(K), α - 2, ch, constraint_noise, solver_bp)
        f_rhs = C * f_inner
        Q_rhs::SparseMatrixCSC{Tv, Ti} = C_inv * sparse(to_matrix(Q_inner.A)) * C_inv
        Q_rhs_sqrt::SparseMatrixCSC{Tv, Ti} = C_inv * to_matrix(Q_inner.A_sqrt)
    end

    apply_soft_constraints!(
        ch,
        constraint_noise;
        K = K,
        f_rhs = f_rhs,
        Q_rhs = Q_rhs,
        Q_rhs_sqrt = Q_rhs_sqrt,
    )
    if any(f_rhs .> 0.)
        μ = lu(K) \ f_rhs
    else
        μ = zeros(Tv, length(f_rhs))
    end
    Q = LinearMaps.WrappedMap{Tv}(Symmetric(K' * (scale_mat * Q_rhs) * K))
    Q_sqrt = LinearMaps.WrappedMap{Tv}(K' * (scale_mat_sqrt * Q_rhs_sqrt))
    return μ, LinearMapWithSqrt(Q, Q_sqrt)
end

"""
    discretize(𝒟::MaternSPDE{D}, discretization::FEMDiscretization{D})::AbstractGMRF where {D}

Discretize a Matérn SPDE using a Finite Element Method (FEM) discretization.
Computes the stiffness and (lumped) mass matrix, and then forms the precision matrix
of the GMRF discretization.
"""
function discretize(
    𝒟::MaternSPDE{D},
    discretization::FEMDiscretization{D};
    solver_blueprint::AbstractSolverBlueprint = DefaultSolverBlueprint(),
)::AbstractGMRF where {D}
    cellvalues = CellValues(
        discretization.quadrature_rule,
        discretization.interpolation,
        discretization.geom_interpolation,
    )
    C̃, G = assemble_C_G_matrices(
        cellvalues,
        discretization.dof_handler,
        discretization.interpolation,
        𝒟.diffusion_factor,
    )
    K = 𝒟.κ^2 * C̃ + G

    # Ratio to get user-specified variance
    ratio = 1.0
    if 𝒟.ν > 0 # TODO: What to do for ν = 0?
        σ²_natural = gamma(𝒟.ν) / (gamma(𝒟.ν + D / 2) * (4π)^(D / 2) * 𝒟.κ^(2 * 𝒟.ν))
        σ²_goal = 𝒟.σ²
        ratio = σ²_natural / σ²_goal
    end

    μ, Q = matern_mean_precision(
        C̃,
        K,
        Integer(α(𝒟)),
        discretization.constraint_handler,
        discretization.constraint_noise,
        solver_blueprint,
        ratio,
    )

    x = GMRF(μ, Q, solver_blueprint)
    return x
end

function range_to_κ(range::Real, ν)
    return √(8ν) / range
end

function smoothness_to_ν(smoothness::Int, D::Int)
    (smoothness >= 0) || throw(ArgumentError("smoothness must be non-negative"))
    return iseven(D) ? smoothness + 1 : (smoothness // 1 + 1 // 2)
end

function product_matern(
    matern_temporal::MaternSPDE,
    N_t::Int,
    matern_spatial::MaternSPDE,
    spatial_disc::FEMDiscretization;
    solver_blueprint = DefaultSolverBlueprint(),
)
    offset = N_t ÷ 10
    temporal_grid = generate_grid(Ferrite.Line, (N_t + 2 * offset - 1,))
    temporal_ip = Lagrange{RefLine,1}()
    temporal_qr = QuadratureRule{RefLine}(2)
    temporal_disc = FEMDiscretization(temporal_grid, temporal_ip, temporal_qr)
    x_t = discretize(matern_temporal, temporal_disc)

    Q_t = to_matrix(precision_map(x_t))[offset+1:end-offset, offset+1:end-offset]
    Q_t = LinearMap(Q_t)
    x_s = discretize(matern_spatial, spatial_disc)
    Q_s = precision_map(x_s)

    return kronecker_product_spatiotemporal_model(
        Q_t,
        Q_s,
        spatial_disc;
        solver_blueprint = solver_blueprint,
    )
end
