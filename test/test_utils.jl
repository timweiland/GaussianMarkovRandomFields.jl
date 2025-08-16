using Ferrite

"""
    _get_periodic_constraint(grid)

Create a periodic constraint for a 1D grid connecting left and right boundaries.
"""
function _get_periodic_constraint(grid)
    cellidx_left, dofidx_left = collect(grid.facetsets["left"])[1]
    cellidx_right, dofidx_right = collect(grid.facetsets["right"])[1]

    temp_dh = DofHandler(grid)
    add!(temp_dh, :u, Lagrange{RefLine, 1}())
    close!(temp_dh)
    cc = CellCache(temp_dh)
    get_dof(cell_idx, dof_idx) = (Ferrite.reinit!(cc, cell_idx); celldofs(cc)[dof_idx])
    dof_left = get_dof(cellidx_left, dofidx_left)
    dof_right = get_dof(cellidx_right, dofidx_right)

    return AffineConstraint(dof_left, [dof_right => 1.0], 0.0)
end

"""
    _get_dirichlet_constraint(grid::Ferrite.Grid{1}, left_val, right_val)

Create a Dirichlet constraint for a 1D grid with specified values at left and right boundaries.
"""
function _get_dirichlet_constraint(grid::Ferrite.Grid{1}, left_val, right_val)
    boundary = getfacetset(grid, "left") ∪ getfacetset(grid, "right")

    return Ferrite.Dirichlet(
        :u,
        boundary,
        x -> (x[1] ≈ -1.0) ? left_val : (x[1] ≈ 1.0) ? right_val : 0.0,
    )
end
