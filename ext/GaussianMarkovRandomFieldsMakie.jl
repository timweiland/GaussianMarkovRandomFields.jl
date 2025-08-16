# COV_EXCL_START
module GaussianMarkovRandomFieldsMakie

using GaussianMarkovRandomFields
using Makie, Random
using Distributions, Tensors

import GaussianMarkovRandomFields: gmrf_fem_1d_plot, gmrf_fem_1d_plot!
import GaussianMarkovRandomFields:
    gmrf_fem_1d_spatiotemporal_plot, gmrf_fem_1d_spatiotemporal_plot!

@recipe(GMRF_FEM_1D_Plot) do scene
    Attributes(
        with_std = true,
        N_samples = 3,
        rng = Random.default_rng(),
        field = :default,
        mean_color = :blue,
        conf_band_color = (:blue, 0.3),
        sample_color = (:gray, 0.3),
    )
end

Makie.plottype(::AbstractGMRF, ::FEMDiscretization{1}) = GMRF_FEM_1D_Plot

function _plot_1d_gaussian!(
        plot_into,
        plot_points,
        means,
        lower,
        upper,
        samples;
        mean_color = :blue,
        conf_band_color = (:blue, 0.3),
        sample_color = (:gray, 0.3),
    )
    lines!(plot_into, plot_points, means, color = mean_color)
    band!(plot_into, plot_points, lower, upper, color = conf_band_color)
    for samp in samples
        lines!(plot_into, plot_points, samp, color = sample_color)
    end
    return
end


function Makie.plot!(
        gmrf_fem_plot::GMRF_FEM_1D_Plot{<:Tuple{<:AbstractGMRF, <:FEMDiscretization{1}}},
    )
    gmrf = gmrf_fem_plot[1]
    disc = gmrf_fem_plot[2]

    means = Observable(mean(gmrf[]))
    stds = Observable(zeros(size(means[])))
    samples = Observable(Vector{Float64}[])
    plot_points = Observable(Float64[])
    confs = @lift(1.96 * $stds)
    upper = Observable(zeros(size(means[])))
    lower = Observable(zeros(size(means[])))
    samples = [Observable(Float64[]) for _ in 1:gmrf_fem_plot.N_samples[]]

    function update_plot(gmrf, disc)
        node_coords = map(n -> n.x[1], disc.grid.nodes)
        bounds = (minimum(node_coords), maximum(node_coords))
        plot_points[] = range(bounds[1], bounds[2], length = 10 * length(node_coords))

        eval_mat = evaluation_matrix(
            disc,
            [Tensors.Vec(x) for x in plot_points[]];
            field = gmrf_fem_plot.field[],
        )

        means[] = eval_mat * mean(gmrf)
        if gmrf_fem_plot.with_std[]
            stds[] = std(gmrf)
        else
            stds[] = zeros(size(mean(gmrf)))
        end
        stds[] = eval_mat * stds[]
        upper[] = means[] + confs[]
        lower[] = means[] - confs[]
        for i in eachindex(samples)
            samples[i][] = eval_mat * rand(gmrf_fem_plot.rng[], gmrf)
        end
        return
    end
    Makie.Observables.onany(update_plot, gmrf, disc)
    update_plot(gmrf[], disc[])

    _plot_1d_gaussian!(
        gmrf_fem_plot,
        plot_points,
        means,
        lower,
        upper,
        samples,
        mean_color = gmrf_fem_plot.mean_color[],
        conf_band_color = gmrf_fem_plot.conf_band_color[],
        sample_color = gmrf_fem_plot.sample_color[],
    )
    return gmrf_fem_plot
end

@recipe(GMRF_FEM_1D_Spatiotemporal_Plot) do scene
    Attributes(
        with_std = true,
        N_samples = 3,
        rng = Random.default_rng(),
        field = :default,
        mean_color = :blue,
        conf_band_color = (:blue, 0.3),
        sample_color = (:gray, 0.3),
    )
end

ST_GMRF_1D = Union{ConstantMeshSTGMRF{1}, LinearConditionalGMRF{<:ConstantMeshSTGMRF{1}}}

function Makie.plot!(
        gmrf_fem_st_plot::GMRF_FEM_1D_Spatiotemporal_Plot{<:Tuple{<:ST_GMRF_1D, <:Int}},
    )
    gmrf = gmrf_fem_st_plot[1]
    t_idx = gmrf_fem_st_plot[2]

    means_t = Observable(time_means(gmrf[]))
    stds_t = Observable(time_stds(gmrf[]))
    samples_t = [
        Observable(time_rands(gmrf[], gmrf_fem_st_plot.rng[])) for
            _ in 1:gmrf_fem_st_plot.N_samples[]
    ]

    means = Observable(Float64[])
    stds = Observable(Float64[])
    samples = [Observable(Float64[]) for _ in 1:gmrf_fem_st_plot.N_samples[]]

    plot_points = Observable(Float64[])
    confs = @lift(1.96 * $stds)
    upper = Observable(zeros(size(means[])))
    lower = Observable(zeros(size(means[])))

    function update_vals(gmrf)
        means_t[] = time_means(gmrf)
        stds_t[] = time_stds(gmrf)
        for i in eachindex(samples)
            samples_t[i][] = time_rands(gmrf, gmrf_fem_st_plot.rng[])
        end
        return update_plot(t_idx[])
    end

    function update_plot(t_idx)
        disc = discretization_at_time(gmrf[], t_idx)
        node_coords = map(n -> n.x[1], disc.grid.nodes)
        bounds = (minimum(node_coords), maximum(node_coords))
        plot_points[] = range(bounds[1], bounds[2], length = 10 * length(node_coords))

        eval_mat = evaluation_matrix(
            disc,
            [Tensors.Vec(x) for x in plot_points[]];
            field = gmrf_fem_st_plot.field[],
        )

        means[] = eval_mat * means_t[][t_idx]
        if gmrf_fem_st_plot.with_std[]
            stds[] = stds_t[][t_idx]
        else
            stds[] = zeros(size(means_t[][t_idx]))
        end
        stds[] = eval_mat * stds[]
        upper[] = means[] + confs[]
        lower[] = means[] - confs[]
        for i in eachindex(samples)
            samples[i][] = eval_mat * samples_t[i][][t_idx]
        end
        return
    end
    Makie.Observables.on(update_plot, t_idx)

    update_vals(gmrf[])

    _plot_1d_gaussian!(
        gmrf_fem_st_plot,
        plot_points,
        means,
        lower,
        upper,
        samples,
        mean_color = gmrf_fem_st_plot.mean_color[],
        conf_band_color = gmrf_fem_st_plot.conf_band_color[],
        sample_color = gmrf_fem_st_plot.sample_color[],
    )
    return gmrf_fem_st_plot
end

Makie.plottype(::ST_GMRF_1D, ::Int) = GMRF_FEM_1D_Spatiotemporal_Plot

end
# COV_EXCL_STOP
