using Ferrite
import Ferrite: ndofs

export FEMDiscretization, ndim, evaluation_matrix, node_selection_matrix

"""
    FEMDiscretization(grid, interpolation, quadrature_rule)

A struct that contains all the information needed to discretize
a (S)PDE using the Finite Element Method.
"""
struct FEMDiscretization{
    D,
    G<:Grid{D},
    I<:Interpolation{D},
    Q<:QuadratureRule{D},
    GI<:Interpolation{D},
    H<:DofHandler{D,G},
}
    grid::G
    interpolation::I
    quadrature_rule::Q
    geom_interpolation::GI
    dof_handler::H

    function FEMDiscretization(
        grid::G,
        interpolation::I,
        quadrature_rule::Q,
    ) where {D,G<:Grid{D},I<:Interpolation{D},Q<:QuadratureRule{D}}
        geom_interpolation = interpolation
        dh = DofHandler(grid)
        add!(dh, :u, 1)
        close!(dh)
        new{D,G,I,Q,I,DofHandler{D,G}}(
            grid,
            interpolation,
            quadrature_rule,
            geom_interpolation,
            dh,
        )
    end

    function FEMDiscretization(
        grid::G,
        interpolation::I,
        quadrature_rule::Q,
        geom_interpolation::GI,
    ) where {D,G<:Grid{D},I<:Interpolation{D},Q<:QuadratureRule{D},GI<:Interpolation{D}}
        dh = DofHandler(grid)
        add!(dh, :u, 1)
        close!(dh)
        new{D,G,I,Q,GI,DofHandler{D,G}}(
            grid,
            interpolation,
            quadrature_rule,
            geom_interpolation,
            dh,
        )
    end
end

"""
    ndim(f::FEMDiscretization)

Return the dimension of space in which the discretization is defined.
Typically ndim(f) == 1, 2, or 3.
"""
ndim(::FEMDiscretization{D}) where {D} = D

"""
    ndofs(f::FEMDiscretization)

Return the number of degrees of freedom in the discretization.
"""
ndofs(f::FEMDiscretization) = f.dof_handler.ndofs.x

"""
    evaluation_matrix(f::FEMDiscretization, X)

Return the matrix A such that A[i, j] is the value of the j-th basis function
at the i-th point in X.
"""
function evaluation_matrix(f::FEMDiscretization{D}, X) where {D}
    A = spzeros(length(X), ndofs(f))
    peh = PointEvalHandler(f.grid, X)
    cc = CellCache(f.dof_handler)
    for i in eachindex(peh.cells)
        reinit!(cc, peh.cells[i])
        dofs = celldofs(cc)
        A[i, [dofs...]] = Ferrite.value(f.interpolation, peh.local_coords[i])
    end
    return A
end

"""
    node_selection_matrix(f::FEMDiscretization, node_ids)

Return the matrix A such that A[i, j] = 1 if the j-th basis function
is associated with the i-th node in node_ids.
"""
function node_selection_matrix(f::FEMDiscretization, node_ids)
    A = spzeros(length(node_ids), ndofs(f))
    node_coords = map(n -> f.grid.nodes[n].x, node_ids)
    peh = PointEvalHandler(f.grid, node_coords)
    cc = CellCache(f.dof_handler)
    for i in eachindex(peh.cells)
        reinit!(cc, peh.cells[i])
        dofs = celldofs(cc)
        coords = getcoordinates(cc)
        for j = 1:length(dofs)
            if coords[j] ≈ node_coords[i]
                A[i, dofs[j]] = 1
            end
        end
    end
    return A
end


function Base.show(io::IO, discretization::FEMDiscretization)
    println(io, "FEMDiscretization")
    println(io, "  grid: ", repr(MIME("text/plain"), discretization.grid))
    println(io, "  interpolation: ", discretization.interpolation)
    println(io, "  quadrature_rule: ", discretization.quadrature_rule)
end
