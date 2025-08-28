
# SPDEs {#SPDEs}
<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.matern_mean_precision' href='#GaussianMarkovRandomFields.matern_mean_precision'><span class="jlbinding">GaussianMarkovRandomFields.matern_mean_precision</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
matern_precision(C_inv::AbstractMatrix, K::AbstractMatrix, α::Integer)
```


Compute the precision matrix of a GMRF discretization of a Matérn SPDE. Implements the recursion described in [[1](/bibliography#Lindgren2011)].

**Arguments**
- `C_inv::AbstractMatrix`: The inverse of the (possibly lumped) mass matrix.
  
- `K::AbstractMatrix`: The stiffness matrix.
  
- `α::Integer`: The parameter α = ν + d/2 of the Matérn SPDE.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/spdes/matern.jl#L96-L106" target="_blank" rel="noreferrer">source</a></Badge>

</details>

