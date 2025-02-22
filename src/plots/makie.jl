# COV_EXCL_START
export gmrf_fem_1d_plot, gmrf_fem_1d_plot!
export gmrf_fem_1d_spatiotemporal_plot, gmrf_fem_1d_spatiotemporal_plot!


### 1D ###

"""
    gmrf_fem_1d_plot(gmrf::AbstractGMRF, disc::FEMDiscretization{1})

Plot the GMRF `gmrf` based on the 1D FEM discretization `disc`.

# Arguments
- `gmrf::AbstractGMRF`: The GMRF to plot.
- `disc::FEMDiscretization{1}`: The FEM discretization.

# Keyword arguments
- `with_std::Bool=true`: Whether to plot the confidence bands.
- `N_samples::Int=3`: The number of samples to plot.
- `rng::AbstractRNG=Random.default_rng()`: The random number generator.
- `field::Symbol=:default`: The field to plot.
- `mean_color`: The color of the mean line.
- `conf_band_color`: The color of the confidence bands.
- `sample_color`: The color of the samples.
"""
function gmrf_fem_1d_plot end
function gmrf_fem_1d_plot! end

"""
    gmrf_fem_1d_spatiotemporal_plot(
        gmrf::Union{
            ConstantMeshSTGMRF{1}, LinearConditionalGMRF{<:ConstantMeshSTGMRF{1}}
            },
        t_idx::Int
    )

Plot the 1D spatiotemporal GMRF `gmrf` at time index `t_idx`.

# Arguments
- `gmrf`: The GMRF to plot.
- `t_idx::Int`: The time index.

# Keyword arguments
- `with_std::Bool=true`: Whether to plot the confidence bands.
- `N_samples::Int=3`: The number of samples to plot.
- `rng::AbstractRNG=Random.default_rng()`: The random number generator.
- `field::Symbol=:default`: The field to plot.
- `mean_color`: The color of the mean line.
- `conf_band_color`: The color of the confidence bands.
- `sample_color`: The color of the samples.
"""
function gmrf_fem_1d_spatiotemporal_plot end
function gmrf_fem_1d_spatiotemporal_plot! end

# COV_EXCL_STOP
