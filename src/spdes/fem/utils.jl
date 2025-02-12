using Ferrite, LinearAlgebra, SparseArrays

export assemble_mass_matrix,
    assemble_diffusion_matrix,
    assemble_advection_matrix,
    lump_matrix,
    assemble_streamline_diffusion_matrix,
    apply_soft_constraints!

"""
    lump_matrix(A::AbstractMatrix, ::Lagrange{D, S, 1}) where {D, S}

Lump a matrix by summing over the rows.
"""
function lump_matrix(A::AbstractMatrix, ::Lagrange{D,S,1}) where {D,S}
    return spdiagm(0 => reshape(sum(A, dims = 2), (size(A)[1],)))
end

"""
    lump_matrix(A::AbstractMatrix, ::Lagrange)

Lump a matrix through HRZ lumping.
Fallback for non-linear elements.
Row-summing cannot be used for non-linear elements, because it does not ensure
positive definiteness.
"""
function lump_matrix(A::AbstractMatrix, ::Lagrange)
    total_mass = sum(A)
    diag_mass = sum(diag(A))
    HRZ_diag = (total_mass / diag_mass) * diag(A)
    return spdiagm(0 => HRZ_diag)
end


function assemble_mass_matrix(
    Ce::SparseMatrixCSC,
    cellvalues::CellValues,
    interpolation;
    lumping = true,
)
    n_basefuncs = getnbasefunctions(cellvalues)
    # Reset to 0
    Ce = spzeros(size(Ce))
    # Loop over quadrature points
    for q_point = 1:getnquadpoints(cellvalues)
        # Get the quadrature weight
        dΩ = getdetJdV(cellvalues, q_point)
        # Loop over test shape functions
        for i = 1:n_basefuncs
            δu = shape_value(cellvalues, q_point, i)
            # Loop over trial shape functions
            for j = 1:n_basefuncs
                u = shape_value(cellvalues, q_point, j)
                # Add contribution to Ce
                Ce[i, j] += (δu ⋅ u) * dΩ
            end
        end
    end
    if lumping
        Ce = lump_matrix(Ce, interpolation)
    end
    return Ce
end

function assemble_diffusion_matrix(
    Ge::SparseMatrixCSC,
    cellvalues::CellValues;
    diffusion_factor = I,
)
    n_basefuncs = getnbasefunctions(cellvalues)
    # Reset to 0
    Ge = spzeros(size(Ge))
    # Loop over quadrature points
    for q_point = 1:getnquadpoints(cellvalues)
        # Get the quadrature weight
        dΩ = getdetJdV(cellvalues, q_point)
        # Loop over test shape functions
        for i = 1:n_basefuncs
            ∇δu = shape_gradient(cellvalues, q_point, i)
            # Loop over trial shape functions
            for j = 1:n_basefuncs
                ∇u = diffusion_factor * shape_gradient(cellvalues, q_point, j)
                # Add contribution to Ke
                Ge[i, j] += (∇δu ⋅ ∇u) * dΩ
            end
        end
    end
    return Ge
end

function assemble_advection_matrix(
    Be::SparseMatrixCSC,
    cellvalues::CellValues;
    advection_velocity = 1,
)
    n_basefuncs = getnbasefunctions(cellvalues)
    # Reset to 0
    Be = spzeros(size(Be))
    # Loop over quadrature points
    for q_point = 1:getnquadpoints(cellvalues)
        # Get the quadrature weight
        dΩ = getdetJdV(cellvalues, q_point)
        # Loop over test shape functions
        for i = 1:n_basefuncs
            ∇δu = shape_gradient(cellvalues, q_point, i)
            # Loop over trial shape functions
            for j = 1:n_basefuncs
                u = shape_value(cellvalues, q_point, j)
                # Add contribution to Ke
                Be[i, j] += (advection_velocity ⋅ ∇δu ⋅ u) * dΩ
            end
        end
    end
    return Be
end

function assemble_streamline_diffusion_matrix(
    Ge::SparseMatrixCSC,
    cellvalues::CellValues,
    advection_velocity,
    h,
)
    n_basefuncs = getnbasefunctions(cellvalues)
    # Reset to 0
    Ge = spzeros(size(Ge))
    # Loop over quadrature points
    for q_point = 1:getnquadpoints(cellvalues)
        # Get the quadrature weight
        dΩ = getdetJdV(cellvalues, q_point)
        # Loop over test shape functions
        for i = 1:n_basefuncs
            ∇δu = shape_gradient(cellvalues, q_point, i)
            # Loop over trial shape functions
            for j = 1:n_basefuncs
                ∇u = shape_gradient(cellvalues, q_point, j)
                # Add contribution to Ke
                Ge[i, j] += ((advection_velocity ⋅ ∇δu) ⋅ (advection_velocity ⋅ ∇u)) * dΩ
            end
        end
    end
    normalization_constant = h * (1 / norm(advection_velocity))
    return normalization_constant * Ge
end

@doc raw"""
    apply_soft_constraints!(K, f_rhs, ch, constraint_noise; Q_rhs = nothing, Q_rhs_sqrt = nothing)

Apply soft constraints to the Gaussian relation
```math
\mathbf{K} \mathbf{u} \sim \mathcal{N}(\mathbf{f}_{\text{rhs}}, \mathbf{Q}_{\text{rhs}}^{-1})
```

Soft means that the constraints are fulfilled up to noise of magnitude specified
by `constraint_noise`.

Modifies `K` and `f_rhs` in place. If `Q_rhs` and `Q_rhs_sqrt` are provided, they
are modified in place as well.

# Arguments
- `K::SparseMatrixCSC`: Stiffness matrix.
- `f_rhs::AbstractVector`: Right-hand side.
- `ch::ConstraintHandler`: Constraint handler.
- `constraint_noise::Vector{Float64}`: Noise for each constraint.
- `Q_rhs::Union{Nothing, SparseMatrixCSC}`: Covariance matrix for the right-hand
                                            side.
- `Q_rhs_sqrt::Union{Nothing, SparseMatrixCSC}`: Square root of the covariance
                                                 matrix for the right-hand side.
"""
function apply_soft_constraints!(
    K::SparseMatrixCSC,
    f_rhs::AbstractVector,
    ch::ConstraintHandler,
    constraint_noise::AbstractVector;
    Q_rhs::Union{Nothing,SparseMatrixCSC} = nothing,
    Q_rhs_sqrt::Union{Nothing,SparseMatrixCSC} = nothing,
)
    for p_dof in ch.prescribed_dofs
        constraint_idx = ch.dofmapping[p_dof]
        r = nzrange(K, p_dof)
        K[p_dof, :] .= 0.0
        dofcoeffs = ch.dofcoefficients[constraint_idx]
        if dofcoeffs !== nothing
            for (k, v) in dofcoeffs
                K[p_dof, k] = -v
                for j in r
                    idx = K.rowval[j]
                    #M[idx, idx] += noise * v
                    if idx != p_dof
                        K[idx, k] += v * K.nzval[j]
                    end
                end
            end
        end
        inhomogeneity = ch.inhomogeneities[constraint_idx]
        if inhomogeneity !== nothing
            f_rhs[p_dof] = inhomogeneity
            for j in r
                f_rhs[K.rowval[j]] -= inhomogeneity * K.nzval[j]
            end
        end
        K.nzval[r] .= 0.0

        K[p_dof, p_dof] = 1.0

        if Q_rhs !== nothing
            Q_rhs[p_dof, :] .= 0.0
            Q_rhs[:, p_dof] .= 0.0
            Q_rhs[p_dof, p_dof] = (constraint_noise[constraint_idx])^(-2)
        end
        if Q_rhs_sqrt !== nothing
            Q_rhs_sqrt[p_dof, :] .= 0.0
            Q_rhs_sqrt[:, p_dof] .= 0.0
            Q_rhs_sqrt[p_dof, p_dof] = constraint_noise[constraint_idx]^(-1)
        end
    end
end
