using Ferrite
using LinearAlgebra
using SparseArrays

export MaternSPDE, α, ndim, discretize, assemble_C_G_matrices, lump_matrix

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
struct MaternSPDE{D} <: SPDE
    κ::Real
    ν::Rational

    function MaternSPDE{D}(κ::Real, ν::Union{Integer,Rational}) where {D}
        κ > 0 || throw(ArgumentError("κ must be positive"))
        ν >= 0 || throw(ArgumentError("ν must be non-negative"))
        (D >= 1 && isinteger(D)) || throw(ArgumentError("D must be a positive integer"))
        new{D}(κ, ν)
    end
end

α(𝒟::MaternSPDE{D}) where {D} = 𝒟.ν + D // 2
ndim(::MaternSPDE{D}) where {D} = D

function assemble_C_G_matrices(cellvalues::CellScalarValues, dh::DofHandler, interpolation)
    C, G = create_sparsity_pattern(dh), create_sparsity_pattern(dh)

    n_basefuncs = getnbasefunctions(cellvalues)
    Ce = spzeros(n_basefuncs, n_basefuncs)
    Ge = spzeros(n_basefuncs, n_basefuncs)

    C_assembler = start_assemble(C)
    G_assembler = start_assemble(G)

    for cell in CellIterator(dh)
        reinit!(cellvalues, cell)
        Ce = assemble_mass_matrix(Ce, cellvalues, interpolation; lumping = true)
        Ge = assemble_diffusion_matrix(Ge, cellvalues)
        assemble!(C_assembler, celldofs(cell), Ce)
        assemble!(G_assembler, celldofs(cell), Ge)
    end
    return C, G
end

"""
    matern_precision(C_inv::AbstractMatrix, K::AbstractMatrix, α::Integer)

Compute the precision matrix of a GMRF discretization of a Matérn SPDE.
Implements the recursion described in [1].

[1] Lindgren, F., Rue, H. and Lindström, J. (2011), An explicit link between
Gaussian fields and Gaussian Markov random fields: the stochastic partial differential
equation approach. Journal of the Royal Statistical Society: Series B
(Statistical Methodology), 73: 423-498.

# Arguments
- `C_inv::AbstractMatrix`: The inverse of the (possibly lumped) mass matrix.
- `K::AbstractMatrix`: The stiffness matrix.
- `α::Integer`: The parameter α = ν + d/2 of the Matérn SPDE.
"""
function matern_precision(C_inv::AbstractMatrix, K::AbstractMatrix, α::Integer)
    if α < 1
        throw(ArgumentError("α must be positive and non-zero"))
    end
    if α == 1
        return K
    elseif α == 2
        return K * C_inv * K
    else
        return K * C_inv * matern_precision(C_inv, K, α - 2) * C_inv * K
    end
end

"""
    discretize(𝒟::MaternSPDE{D}, discretization::FEMDiscretization{D})::AbstractGMRF where {D}

Discretize a Matérn SPDE using a Finite Element Method (FEM) discretization.
Computes the stiffness and (lumped) mass matrix, and then forms the precision matrix
of the GMRF discretization.
"""
function discretize(
    𝒟::MaternSPDE{D},
    discretization::FEMDiscretization{D},
)::AbstractGMRF where {D}
    cellvalues =
        CellScalarValues(discretization.quadrature_rule, discretization.interpolation)
    C̃, G = assemble_C_G_matrices(
        cellvalues,
        discretization.dof_handler,
        discretization.interpolation,
    )
    K = 𝒟.κ^2 * C̃ + G
    C̃⁻¹ = spdiagm(0 => 1 ./ diag(C̃))
    Q = matern_precision(C̃⁻¹, K, Integer(α(𝒟)))
    Q = (Q + Q') / 2 # Ensure symmetry. TODO: Can this be guaranteed naturally?
    return GMRF(spzeros(size(Q, 1)), Q)
end
