using Ferrite, Makie

export plot_spde_gmrf, plot_spatiotemporal_gmrf, plot_spatiotemporal_gmrf_at_time

# COV_EXCL_START

### 1D ###

function _plot_1d_gaussian!(
    ax,
    plot_points,
    means,
    lower,
    upper,
    samples;
    mean_color = :blue,
    conf_band_color = (:blue, 0.3),
    sample_color = (:gray, 0.3),
)
    lines!(ax, plot_points, means, color = mean_color)
    band!(ax, plot_points, lower, upper, color = conf_band_color)
    for samp in samples
        lines!(ax, plot_points, samp, color = sample_color)
    end
end

function plot_spde_gmrf(
    x::AbstractGMRF,
    disc::FEMDiscretization{1};
    combined = true,
    mean_pos = (1, 1),
    std_pos = (1, 2),
    sample_pos = [(2, 1), (2, 2)],
    N_samples = 3,
    limits = nothing,
    compute_std = true,
    field = :default,
    rng = Random.default_rng(),
)
    if combined
        return _plot_spde_gmrf_combined(
            x,
            disc,
            limits = limits,
            compute_std = compute_std,
            N_samples = N_samples,
            field = field,
            rng = rng,
        )
    else
        return _plot_spde_gmrf_separate(
            x,
            disc,
            mean_pos = mean_pos,
            std_pos = std_pos,
            sample_pos = sample_pos,
            limits = limits,
            compute_std = compute_std,
            field = field,
            rng = rng,
        )
    end
end

function _plot_spde_gmrf_separate(
    d::AbstractGMRF,
    disc::FEMDiscretization{1};
    mean_pos = (1, 1),
    std_pos = (1, 2),
    sample_pos = [(2, 1), (2, 2)],
    limits = nothing,
    compute_std = true,
    field = :default,
    rng = Random.default_rng(),
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
    samples = [rand(rng, d) for _ in sample_pos]
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

function _plot_spde_gmrf_combined(
    d::AbstractGMRF,
    disc::FEMDiscretization{1};
    limits = nothing,
    compute_std = true,
    field = :default,
    N_samples = 3,
    rng = Random.default_rng(),
)
    node_coords = map(n -> n.x[1], disc.grid.nodes)
    bounds = (minimum(node_coords), maximum(node_coords))
    plot_points = range(bounds[1], bounds[2], length = 10 * length(node_coords))

    eval_mat = evaluation_matrix(disc, [Tensors.Vec(x) for x in plot_points]; field = field)

    means = eval_mat * mean(d)
    if compute_std
        stds = std(d)
    else
        stds = spzeros(size(means))
    end
    stds = eval_mat * stds
    samples = [eval_mat * rand(rng, d) for _ = 1:N_samples]

    confs = 1.96 * stds
    upper = means + confs
    lower = means - confs

    if limits === nothing
        limits = (nothing, nothing)
    end
    fig = Figure()
    ax = Axis(fig[1, 1]; limits = limits)
    _plot_1d_gaussian!(ax, plot_points, means, lower, upper, samples)
    return fig
end

ST_GMRF_1D = Union{
    ConstantMeshSTGMRF{1},
    LinearConditionalGMRF{<:ConstantMeshSTGMRF{1}},
    ConstrainedGMRF{<:ConstantMeshSTGMRF{1}},
    ConstrainedGMRF{<:LinearConditionalGMRF{<:ConstantMeshSTGMRF{1}}},
}

### 1D ###
function plot_spatiotemporal_gmrf_at_time(
    x::ST_GMRF_1D,
    t::Int;
    limits = nothing,
    compute_std = true,
    N_samples = 3,
    field = :default,
    rng = Random.default_rng(),
)
    disc = discretization_at_time(x, t)

    node_coords = map(n -> n.x[1], disc.grid.nodes)
    bounds = (minimum(node_coords), maximum(node_coords))
    plot_points = range(bounds[1], bounds[2], length = 10 * length(node_coords))

    eval_mat = evaluation_matrix(disc, [Tensors.Vec(x) for x in plot_points]; field = field)

    means = time_means(x)[t]
    if compute_std
        stds = time_stds(x)[t]
    else
        stds = spzeros(size(means))
    end
    means = eval_mat * means
    stds = eval_mat * stds
    samples = [eval_mat * time_rands(x, rng)[t] for _ = 1:N_samples]
    confs = 1.96 * stds
    upper = means + confs
    lower = means - confs

    if limits === nothing
        limits = (nothing, nothing)
    end
    fig = Figure()
    ax = Axis(fig[1, 1]; limits = limits)
    _plot_1d_gaussian!(ax, plot_points, means, lower, upper, samples)
    return fig
end

function _plot_spatiotemporal_gmrf_separate(
    x::ST_GMRF_1D;
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
    x::ST_GMRF_1D;
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
    _plot_1d_gaussian!(ax, plot_points, cur_means, lower, upper, cur_samples)

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
    x::ST_GMRF_1D;
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
