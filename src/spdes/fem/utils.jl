using Ferrite, LinearAlgebra, SparseArrays

export assemble_mass_matrix, assemble_diffusion_matrix, assemble_advection_matrix, 
        lump_matrix, assemble_streamline_diffusion_matrix

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


function assemble_mass_matrix(Ce::SparseMatrixCSC, cellvalues::CellScalarValues, interpolation; lumping = true)
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

function assemble_diffusion_matrix(Ge::SparseMatrixCSC, cellvalues::CellScalarValues; diffusion_factor=I)
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
