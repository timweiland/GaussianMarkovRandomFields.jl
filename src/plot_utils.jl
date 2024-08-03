using Ferrite, FerriteViz, WGLMakie
import GeometryBasics

export plot_spde_gmrf, get_coords, in_bounds, in_bounds_cell, hide_triangles

"""
    get_coords(grid, cell_idx)

Get the coordinates of the nodes of a cell in the grid.
"""
function get_coords(grid, cell_idx)
    return [grid.nodes[n].x for n in grid.cells[cell_idx].nodes]
end

"""
    in_bounds(coord, bounds)

Check if a coordinate is within the bounding box defined by `bounds`.
`bounds` is a list of pairs of the form (lower_bound, upper_bound) of the same
length as `coord`.
"""
function in_bounds(coord, bounds)
    if length(bounds) != length(coord)
        throw(ArgumentError("Bounds must have the same length as the coordinate"))
    end
    return all([
        coord[i] >= bounds[i][1] && coord[i] <= bounds[i][2] for i = 1:length(coord)
    ])
end

"""
    in_bounds_cell(grid, bounds, cell_idx)

Check if a cell is within the bounding box defined by `bounds`.
"""
function in_bounds_cell(grid, bounds, cell_idx)
    return all([in_bounds(coord, bounds) for coord in get_coords(grid, cell_idx)])
end

# COV_EXCL_START
"""
    hide_triangles(mp, vis_cells)

Hide the triangles in a MakiePlotter object that correspond to cells that 
are not in `vis_cells`.
"""
function hide_triangles(mp::MakiePlotter{2,DH,T}, vis_cells) where {DH,T}
    vis_triangles = copy(mp.vis_triangles)
    vis_triangles[vis_cells[mp.triangle_cell_map]] =
        mp.all_triangles[vis_cells[mp.triangle_cell_map]]
    vis_triangles[.!vis_cells[mp.triangle_cell_map]] = [
        GeometryBasics.GLTriangleFace(1, 1, 1) for
        i = 1:sum(.!vis_cells[mp.triangle_cell_map])
    ]
    return MakiePlotter{
        2,
        DH,
        T,
        typeof(mp.topology),
        Float32,
        typeof(mp.mesh),
        eltype(mp.vis_triangles),
    }(
        mp.dh,
        mp.u,
        mp.topology,
        vis_cells,
        mp.gridnodes,
        mp.physical_coords,
        mp.physical_coords_mesh,
        mp.all_triangles,
        vis_triangles,
        mp.triangle_cell_map,
        mp.cell_triangle_offsets,
        mp.reference_coords,
        mp.mesh,
    )
end

"""
    plot_spde_gmrf(d::GMRF, disc::FEMDiscretization; mean_pos, std_pos, sample_pos,
    plot_surface, limits, compute_std)

Plot the mean, standard deviation, and samples of a GMRF derived from a FEM
discretization of an SPDE.
"""
function plot_spde_gmrf(
    d::GMRF,
    disc::FEMDiscretization;
    mean_pos = (1, 1),
    std_pos = (1, 2),
    sample_pos = [(2, 1), (2, 2)],
    plot_surface = false,
    limits = nothing,
    compute_std = true,
)
    means = mean(d)
    if compute_std
        stds = std(d)
    else
        stds = spzeros(size(means))
    end
    samples = rand(d, (length(sample_pos),))

    if limits === nothing
        limits = plot_surface ? (nothing, nothing, nothing) : (nothing, nothing)
    end
    vis_cells = nothing
    if !any(map(x -> x === nothing, limits))
        vis_cells = map(
            c -> in_bounds_cell(
                disc.grid,
                ((limits[1], limits[2]), (limits[3], limits[4])),
                c,
            ),
            eachindex(disc.grid.cells),
        )
    end
    function plot_fn(mp)
        if vis_cells !== nothing
            mp = hide_triangles(mp, vis_cells)
        end
        if plot_surface
            FerriteViz.surface!(mp)
        else
            FerriteViz.solutionplot!(mp)
        end
    end
    axis_fn = plot_surface ? Axis3 : Axis
    dh = disc.dof_handler

    fig = Figure()
    axis_fn(fig[mean_pos...]; limits = limits)
    mean_plotter = MakiePlotter(dh, Array(means))
    plot_fn(mean_plotter)
    axis_fn(fig[std_pos...]; limits = limits)
    mp = MakiePlotter(dh, Array(stds))
    plot_fn(mp)
    for (i, pos) in enumerate(sample_pos)
        axis_fn(fig[pos...]; limits = limits)
        mp = MakiePlotter(dh, samples[i])
        plot_fn(mp)
    end
    return fig
end
# COV_EXCL_STOP
