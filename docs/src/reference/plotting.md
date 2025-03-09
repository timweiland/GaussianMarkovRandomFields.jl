# Plotting

GaussianMarkovRandomFields.jl offers recipes to make plotting spatial and spatiotemporal GMRFs effortless.
These recipes are contained in a package extension which gets loaded automatically when using Makie (or rather: one of its backends).

All of the following methods may also be called more simply by calling `plot`
or `plot!` with appropriate arguments.

!!! note
    Currently, we only provide recipes for 1D spatial and spatiotemporal GMRFs.
    This may change soon (feel free to open a PR!)
    Until then, note that any 2D or 3D FEM-based GMRF may be plotted indirectly
    through Ferrite's support of VTK files, which may then subsequently be
    opened e.g. in ParaView for visualization.
    See [Spatial Modelling with SPDEs](@ref) for an example.
    

```@docs
gmrf_fem_1d_plot
gmrf_fem_1d_spatiotemporal_plot
```
