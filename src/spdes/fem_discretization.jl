import Ferrite:
    Grid,
    Interpolation,
    QuadratureRule,
    DofHandler,
    add!,
    close!,
    PointEvalHandler,
    ndofs,
    value

export FEMDiscretization, ndim, evaluation_matrix

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
    H<:DofHandler{D,G},
}
    grid::G
    interpolation::I
    quadrature_rule::Q
    dof_handler::H

    function FEMDiscretization(
        grid::G,
        interpolation::I,
        quadrature_rule::Q,
    ) where {D,G<:Grid{D},I<:Interpolation{D},Q<:QuadratureRule{D}}
        dh = DofHandler(grid)
        add!(dh, :u, 1)
        close!(dh)
        new{D,G,I,Q,DofHandler{D,G}}(grid, interpolation, quadrature_rule, dh)
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
    for i in eachindex(peh.cells)
        # A[i, j] is non-zero iff node j is in the cell that contains X[i]
        nodes = f.grid.cells[peh.cells[i]].nodes
        A[i, [nodes...]] = value(f.interpolation, peh.local_coords[i])
    end
    return A
end


function Base.show(io::IO, discretization::FEMDiscretization)
    println(io, "FEMDiscretization")
    println(io, "  grid: ", repr(MIME("text/plain"), discretization.grid))
    println(io, "  interpolation: ", discretization.interpolation)
    println(io, "  quadrature_rule: ", discretization.quadrature_rule)
end
