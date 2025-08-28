
# Discretizations {#Discretizations}

## Spatial discretization with FEM {#Spatial-discretization-with-FEM}
<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.shape_gradient_local' href='#GaussianMarkovRandomFields.shape_gradient_local'><span class="jlbinding">GaussianMarkovRandomFields.shape_gradient_local</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
shape_gradient_local(f::FEMDiscretization, shape_idx::Int, ξ)
```


Gradient of the shape function with index `shape_idx` with respect to the local coordinates `ξ`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/spdes/fem/fem_derivatives.jl#L6-L11" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.shape_gradient_global' href='#GaussianMarkovRandomFields.shape_gradient_global'><span class="jlbinding">GaussianMarkovRandomFields.shape_gradient_global</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
shape_gradient_global(f::FEMDiscretization, dof_coords, shape_idx::Int, ξ; J⁻¹ = nothing)
```


Gradient of the shape function with index `shape_idx` in a cell with node coordinates `dof_coords`, taken with respect to the global coordinates but computed in terms of the local coordinates `ξ`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/spdes/fem/fem_derivatives.jl#L64-L70" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.shape_hessian_local' href='#GaussianMarkovRandomFields.shape_hessian_local'><span class="jlbinding">GaussianMarkovRandomFields.shape_hessian_local</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
shape_hessian_local(f::FEMDiscretization, shape_idx::Int, ξ)
```


Hessian of the shape function with index `shape_idx` with respect to the local coordinates `ξ`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/spdes/fem/fem_derivatives.jl#L19-L24" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.shape_hessian_global' href='#GaussianMarkovRandomFields.shape_hessian_global'><span class="jlbinding">GaussianMarkovRandomFields.shape_hessian_global</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
shape_hessian_global(f::FEMDiscretization, dof_coords, shape_idx::Int, ξ; J⁻¹ = nothing, geo_hessian = nothing)
```


Hessian of the shape function with index `shape_idx` in a cell with node coordinates `dof_coords`, taken with respect to the global coordinates but computed in terms of the local coordinates `ξ`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/spdes/fem/fem_derivatives.jl#L84-L90" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.geom_jacobian' href='#GaussianMarkovRandomFields.geom_jacobian'><span class="jlbinding">GaussianMarkovRandomFields.geom_jacobian</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
geom_jacobian(f::FEMDiscretization, dof_coords, ξ)
```


Jacobian of the geometry mapping at the local coordinates `ξ` with node coordinates `dof_coords`. By &quot;geometry mapping&quot;, we mean the mapping from the reference element to the physical element.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/spdes/fem/fem_derivatives.jl#L32-L39" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.geom_hessian' href='#GaussianMarkovRandomFields.geom_hessian'><span class="jlbinding">GaussianMarkovRandomFields.geom_hessian</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
geom_hessian(f::FEMDiscretization, dof_coords, ξ)
```


Hessian of the geometry mapping at the local coordinates `ξ` with node coordinates `dof_coords`. By &quot;geometry mapping&quot;, we mean the mapping from the reference element to the physical element.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/spdes/fem/fem_derivatives.jl#L49-L56" target="_blank" rel="noreferrer">source</a></Badge>

</details>

