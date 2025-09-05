# Spatial Utilities

Helpers for constructing spatial structures frequently used in latent models.

## Contiguity Adjacency from Polygons

Build a binary adjacency matrix for polygon features using queen contiguity
(shared boundary point). Accepts either a vector of geometries or a LibGEOS
`GeometryCollection`. A Shapefile convenience method is provided via package
extension. It requires you to load Shapefile.jl first.

### Example

```julia
using GaussianMarkovRandomFields, LibGEOS
g1 = readgeom("POLYGON((0 0,1 0,1 1,0 1,0 0))")
g2 = readgeom("POLYGON((1 0,2 0,2 1,1 1,1 0))")
W  = contiguity_adjacency([g1, g2])  # 2×2 symmetric, ones off-diagonal
```

```@docs
GaussianMarkovRandomFields.contiguity_adjacency
```

## See Also

- The BYM + fixed effects Poisson tutorial uses this to build `W` from a shapefile: [Advanced GMRF modelling for disease mapping](@ref)

