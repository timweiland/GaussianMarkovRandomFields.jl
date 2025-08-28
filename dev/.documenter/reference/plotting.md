
# Plotting {#Plotting}

GaussianMarkovRandomFields.jl offers recipes to make plotting spatial and spatiotemporal GMRFs effortless. These recipes are contained in a package extension which gets loaded automatically when using Makie (or rather: one of its backends).

All of the following methods may also be called more simply by calling `plot` or `plot!` with appropriate arguments.

::: tip Note

Currently, we only provide recipes for 1D spatial and spatiotemporal GMRFs. This may change soon (feel free to open a PR!) Until then, note that any 2D or 3D FEM-based GMRF may be plotted indirectly through Ferrite&#39;s support of VTK files, which may then subsequently be opened e.g. in ParaView for visualization. See [Spatial Modelling with SPDEs](/tutorials/spatial_modelling_spdes#Spatial-Modelling-with-SPDEs) for an example.

:::
<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.gmrf_fem_1d_plot' href='#GaussianMarkovRandomFields.gmrf_fem_1d_plot'><span class="jlbinding">GaussianMarkovRandomFields.gmrf_fem_1d_plot</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
gmrf_fem_1d_plot(gmrf::AbstractGMRF, disc::FEMDiscretization{1})
```


Plot the GMRF `gmrf` based on the 1D FEM discretization `disc`.

**Arguments**
- `gmrf::AbstractGMRF`: The GMRF to plot.
  
- `disc::FEMDiscretization{1}`: The FEM discretization.
  

**Keyword arguments**
- `with_std::Bool=true`: Whether to plot the confidence bands.
  
- `N_samples::Int=3`: The number of samples to plot.
  
- `rng::AbstractRNG=Random.default_rng()`: The random number generator.
  
- `field::Symbol=:default`: The field to plot.
  
- `mean_color`: The color of the mean line.
  
- `conf_band_color`: The color of the confidence bands.
  
- `sample_color`: The color of the samples.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/plots/makie.jl#L8-L25" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.gmrf_fem_1d_spatiotemporal_plot' href='#GaussianMarkovRandomFields.gmrf_fem_1d_spatiotemporal_plot'><span class="jlbinding">GaussianMarkovRandomFields.gmrf_fem_1d_spatiotemporal_plot</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
gmrf_fem_1d_spatiotemporal_plot(
    gmrf::Union{
        ConstantMeshSTGMRF{1}, LinearConditionalGMRF{<:ConstantMeshSTGMRF{1}}
        },
    t_idx::Int
)
```


Plot the 1D spatiotemporal GMRF `gmrf` at time index `t_idx`.

**Arguments**
- `gmrf`: The GMRF to plot.
  
- `t_idx::Int`: The time index.
  

**Keyword arguments**
- `with_std::Bool=true`: Whether to plot the confidence bands.
  
- `N_samples::Int=3`: The number of samples to plot.
  
- `rng::AbstractRNG=Random.default_rng()`: The random number generator.
  
- `field::Symbol=:default`: The field to plot.
  
- `mean_color`: The color of the mean line.
  
- `conf_band_color`: The color of the confidence bands.
  
- `sample_color`: The color of the samples.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/plots/makie.jl#L29-L51" target="_blank" rel="noreferrer">source</a></Badge>

</details>

