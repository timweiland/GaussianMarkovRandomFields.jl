
# Meshes {#Meshes}

Finite element method discretizations of SPDEs require a mesh. GaussianMarkovRandomFields provides some utility functions to create meshes for common use cases.

For a hands-on example meshing a 2D point cloud, check out the tutorial [Spatial Modelling with SPDEs](/tutorials/spatial_modelling_spdes#Spatial-Modelling-with-SPDEs).
<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.generate_mesh' href='#GaussianMarkovRandomFields.generate_mesh'><span class="jlbinding">GaussianMarkovRandomFields.generate_mesh</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
generate_mesh(mp::GeometryBasics.MultiPoint, buffer_width::Real,
              interior_mesh_size::Real;
              exterior_mesh_size::Real = 2 * interior_mesh_size,
              element_order::Int = 1, save_path=nothing)
```


Generate a mesh for a spatial point cloud, with a buffer to counteract boundary effects from the SPDE discretization.

**Input**
- `mp` – MultiPoint object
  
- `buffer_width` – Width of the buffer around the convex hull
  
- `interior_mesh_size` – Mesh size inside the convex hull
  
- `exterior_mesh_size` – Mesh size outside the convex hull
  
- `element_order` – Order of the element basis functions
  
- `save_path` – Optional path to save the mesh
  

**Output**

A `Ferrite.Grid` object

**Algorithm**
1. Create the convex hull of the point cloud via LibGEOS
  
2. Create the buffer around the convex hull
  
3. Create a mesh for the buffered polygon using Gmsh
  
4. Transfer the Gmsh information to Ferrite
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/mesh/scattered.jl#L22-L50" target="_blank" rel="noreferrer">source</a></Badge>



```julia
generate_mesh(points, buffer_width::Real, interior_mesh_size::Real;
              exterior_mesh_size::Real=2 * interior_mesh_size,
              element_order::Int=1, save_path=nothing)
```


Generate a mesh from a list of points using Gmsh.

**Input**
- `points` – List of points
  
- `buffer_width` – Width of the buffer around the convex hull
  
- `interior_mesh_size` – Mesh size inside the convex hull
  
- `exterior_mesh_size` – Mesh size outside the convex hull
  
- `element_order` – Order of the elements
  
- `save_path` – Path to save the mesh
  

**Output**

A Ferrite.Grid object


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/mesh/scattered.jl#L149-L168" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.create_inflated_rectangle' href='#GaussianMarkovRandomFields.create_inflated_rectangle'><span class="jlbinding">GaussianMarkovRandomFields.create_inflated_rectangle</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
create_inflated_rectangle(x0, y0, dx, dy, boundary_width, interior_mesh_size,
exterior_mesh_size = 2 * interior_mesh_size; element_order = 1)
```


Create a triangular FEM discretization of a rectangle with an inflated boundary. Useful for FEM discretizations of SPDEs, where the domain is often artificially inflated to avoid undesirable boundary effects. Mesh has physical groups &quot;Domain&quot;, &quot;Interior&quot;, &quot;Interior boundary&quot; and possibly &quot;Exterior boundary&quot;.

**Arguments**
- `x0::Real`: x-coordinate of the bottom-left corner of the rectangle
  
- `y0::Real`: y-coordinate of the bottom-left corner of the rectangle
  
- `dx::Real`: Width of the rectangle
  
- `dy::Real`: Height of the rectangle
  
- `boundary_width::Real`: Width of the inflated boundary. If 0.0, mesh will not                         be inflated at all.
  
- `interior_mesh_size::Real`: Mesh size in the interior of the rectangle
  
- `exterior_mesh_size::Real`: Mesh size in the exterior of the rectangle
  
- `element_order::Int`: Order of the FEM elements
  

**Returns**
- `grid::Ferrite.Grid`: the FEM discretization of the rectangle
  
- `boundary_tags::Vector{Int}`: the indices of the boundary nodes
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/mesh/rectangle.jl#L5-L30" target="_blank" rel="noreferrer">source</a></Badge>

</details>

