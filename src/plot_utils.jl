using Ferrite, FerriteViz, Makie
import GeometryBasics

export plot_spde_gmrf,
    get_coords, in_bounds, in_bounds_cell, hide_triangles, plot_spatiotemporal_gmrf

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
    field = :default,
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
            FerriteViz.surface!(mp; field = field)
        else
            FerriteViz.solutionplot!(mp; field = field)
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

### 1D ###
function plot_spde_gmrf(
    d::AbstractGMRF,
    disc::FEMDiscretization{1};
    mean_pos = (1, 1),
    std_pos = (1, 2),
    sample_pos = [(2, 1), (2, 2)],
    limits = nothing,
    compute_std = true,
    field = :default,
)
    node_coords = map(n -> n.x[1], disc.grid.nodes)
    bounds = (minimum(node_coords), maximum(node_coords))
    plot_points = range(bounds[1], bounds[2], length = 10 * length(node_coords))

    eval_mat = evaluation_matrix(disc, [Tensors.Vec(x) for x in plot_points]; field = field)

    means = full_mean(d)
    if compute_std
        stds = full_std(d)
    else
        stds = spzeros(size(means))
    end
    means = eval_mat * means
    stds = eval_mat * stds
    samples = [full_rand(Random.default_rng(), d) for _ in sample_pos]
    samples = [eval_mat * s for s in samples]

    if limits === nothing
        limits = (nothing, nothing)
    end
    fig = Figure()
    mean_ax = Axis(fig[mean_pos...]; limits = limits)
    lines!(mean_ax, plot_points, means)
    std_ax = Axis(fig[std_pos...]; limits = limits)
    lines!(std_ax, plot_points, stds)
    for (i, pos) in enumerate(sample_pos)
        ax = Axis(fig[pos...]; limits = limits)
        lines!(ax, plot_points, samples[i])
    end
    return fig
end


function plot_spatiotemporal_gmrf(
    x::Union{ConstantMeshSTGMRF{2},LinearConditionalGMRF{<:ConstantMeshSTGMRF{2}}};
    mean_pos = (1, 1),
    std_pos = (1, 2),
    sample_pos = [(2, 1), (2, 2)],
    slider_row = 3,
    plot_surface = false,
    limits = nothing,
    compute_std = true,
    field = :default,
)
    disc = discretization_at_time(x, 1)
    means = time_means(x)
    if compute_std
        stds = time_stds(x)
    else
        stds = [spzeros(size(m)) for m in means]
    end
    rng = Random.default_rng()
    samples = [time_rands(x, rng) for _ in sample_pos]

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
            FerriteViz.surface!(mp; field = field)
        else
            FerriteViz.solutionplot!(mp; field = field)
        end
    end
    axis_fn = plot_surface ? Axis3 : Axis
    dh = disc.dof_handler

    fig = Figure()
    axis_fn(fig[mean_pos...]; limits = limits)
    mp_mean = MakiePlotter(dh, Array(means[1]))
    plot_fn(mp_mean)
    axis_fn(fig[std_pos...]; limits = limits)
    mp_std = MakiePlotter(dh, Array(stds[1]))
    plot_fn(mp_std)
    mp_samples = [MakiePlotter(dh, Array(sample[1])) for sample in samples]
    for (i, pos) in enumerate(sample_pos)
        axis_fn(fig[pos...]; limits = limits)
        plot_fn(mp_samples[i])
    end

    time_slider =
        Makie.Slider(fig[slider_row, 1:2], range = 1:length(means), startvalue = 1)
    on(time_slider.value) do val
        FerriteViz.update!(mp_mean, Array(means[val]))
        FerriteViz.update!(mp_std, Array(stds[val]))
        for (i, sample) in enumerate(samples)
            FerriteViz.update!(mp_samples[i], Array(sample[val]))
        end
    end
    return fig
end

### 1D ###
function _plot_spatiotemporal_gmrf_separate(
    x::Union{ConstantMeshSTGMRF{1},LinearConditionalGMRF{<:ConstantMeshSTGMRF{1}}};
    mean_pos = (1, 1),
    std_pos = (1, 2),
    sample_pos = [(2, 1), (2, 2)],
    slider_row = 3,
    limits = nothing,
    compute_std = true,
    field = :default,
)
    disc = discretization_at_time(x, 1)

    node_coords = map(n -> n.x[1], disc.grid.nodes)
    bounds = (minimum(node_coords), maximum(node_coords))
    plot_points = range(bounds[1], bounds[2], length = 10 * length(node_coords))

    eval_mat = evaluation_matrix(disc, [Tensors.Vec(x) for x in plot_points]; field = field)

    means = time_means(x)
    if compute_std
        stds = time_stds(x)
    else
        stds = [spzeros(size(m)) for m in means]
    end
    rng = Random.default_rng()
    samples = [time_rands(x, rng) for _ in sample_pos]

    means = [eval_mat * Array(m) for m in means]
    cur_means = Observable(means[1])
    stds = [eval_mat * Array(s) for s in stds]
    cur_stds = Observable(stds[1])
    samples = [[eval_mat * Array(s) for s in sample] for sample in samples]
    cur_samples = [Observable(sample[1]) for sample in samples]

    if limits === nothing
        limits = (nothing, nothing)
    end
    fig = Figure()
    mean_ax = Axis(fig[mean_pos...]; limits = limits)
    lines!(mean_ax, plot_points, cur_means)
    std_ax = Axis(fig[std_pos...]; limits = limits)
    lines!(std_ax, plot_points, cur_stds)
    sample_axs = [Axis(fig[pos...]; limits = limits) for pos in sample_pos]
    for (i, pos) in enumerate(sample_pos)
        lines!(sample_axs[i], plot_points, cur_samples[i])
    end

    time_slider =
        Makie.Slider(fig[slider_row, 1:2], range = 1:length(means), startvalue = 1)
    on(time_slider.value) do val
        cur_means[] = means[val]
        cur_stds[] = stds[val]
        for (i, sample) in enumerate(samples)
            cur_samples[i][] = sample[val]
        end
    end
    return fig
end

function _plot_spatiotemporal_gmrf_combined(
    x::Union{ConstantMeshSTGMRF{1},LinearConditionalGMRF{<:ConstantMeshSTGMRF{1}}};
    limits = nothing,
    compute_std = true,
    N_samples = N_samples,
    field = :default,
)
    disc = discretization_at_time(x, 1)

    node_coords = map(n -> n.x[1], disc.grid.nodes)
    bounds = (minimum(node_coords), maximum(node_coords))
    plot_points = range(bounds[1], bounds[2], length = 10 * length(node_coords))

    eval_mat = evaluation_matrix(disc, [Tensors.Vec(x) for x in plot_points]; field = field)

    means = time_means(x)
    if compute_std
        stds = time_stds(x)
    else
        stds = [spzeros(size(m)) for m in means]
    end
    rng = Random.default_rng()
    samples = [time_rands(x, rng) for _ = 1:N_samples]

    means = [eval_mat * Array(m) for m in means]
    cur_means = Observable(means[1])
    stds = [eval_mat * Array(s) for s in stds]
    cur_stds = Observable(stds[1])
    samples = [[eval_mat * Array(s) for s in sample] for sample in samples]
    cur_confs = @lift(1.96 * $cur_stds)
    upper = @lift($cur_means + $cur_confs)
    lower = @lift($cur_means - $cur_confs)
    cur_samples = [Observable(sample[1]) for sample in samples]

    if limits === nothing
        limits = (nothing, nothing)
    end
    fig = Figure()
    ax = Axis(fig[1, 1]; limits = limits)
    lines!(ax, plot_points, cur_means, color = :blue)
    band!(ax, plot_points, lower, upper, color = :blue, alpha = 0.3)
    for (i, sample) in enumerate(samples)
        lines!(ax, plot_points, cur_samples[i], color = :gray, alpha = 0.3)
    end

    time_slider = Makie.Slider(fig[2, 1], range = 1:length(means), startvalue = 1)
    on(time_slider.value) do val
        cur_means[] = means[val]
        cur_stds[] = stds[val]
        for (i, sample) in enumerate(samples)
            cur_samples[i][] = sample[val]
        end
    end
    return fig
end


function plot_spatiotemporal_gmrf(
    x::Union{ConstantMeshSTGMRF{1},LinearConditionalGMRF{<:ConstantMeshSTGMRF{1}}};
    combined = true,
    mean_pos = (1, 1),
    std_pos = (1, 2),
    sample_pos = [(2, 1), (2, 2)],
    N_samples = 2,
    slider_row = 3,
    limits = nothing,
    compute_std = true,
    field = :default,
)
    if combined
        return _plot_spatiotemporal_gmrf_combined(
            x,
            limits = limits,
            compute_std = compute_std,
            N_samples = N_samples,
            field = field,
        )
    else
        return _plot_spatiotemporal_gmrf_separate(
            x,
            mean_pos = mean_pos,
            std_pos = std_pos,
            sample_pos = sample_pos,
            slider_row = slider_row,
            limits = limits,
            compute_std = compute_std,
            field = field,
        )
    end
end


# COV_EXCL_STOP
