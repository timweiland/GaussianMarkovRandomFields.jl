
# GMRFs {#GMRFs}
<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.AbstractGMRF' href='#GaussianMarkovRandomFields.AbstractGMRF'><span class="jlbinding">GaussianMarkovRandomFields.AbstractGMRF</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
AbstractGMRF
```


A [Gaussian Markov Random Field](https://en.wikipedia.org/wiki/Markov_random_field#Gaussian)  (GMRF) is a special case of a multivariate normal distribution where the precision matrix is sparse. The zero entries in the precision correspond to conditional independencies.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/gmrf.jl#L59-L65" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.GMRF' href='#GaussianMarkovRandomFields.GMRF'><span class="jlbinding">GaussianMarkovRandomFields.GMRF</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
GMRF(mean, precision, alg=LinearSolve.DefaultLinearSolver(); Q_sqrt=nothing, rbmc_strategy=RBMCStrategy(1000), linsolve_cache=nothing)
```


A Gaussian Markov Random Field with mean `mean` and precision matrix `precision`.

**Arguments**
- `mean::AbstractVector`: The mean vector of the GMRF.
  
- `precision::Union{LinearMap, AbstractMatrix}`: The precision matrix (inverse covariance) of the GMRF.
  
- `alg`: LinearSolve algorithm to use for linear system solving. Defaults to `LinearSolve.DefaultLinearSolver()`.
  
- `Q_sqrt::Union{Nothing, AbstractMatrix}`: Square root of precision matrix Q, used for sampling when algorithm doesn&#39;t support backward solve.
  
- `rbmc_strategy`: RBMC strategy for marginal variance computation when selected inversion is unavailable. Defaults to `RBMCStrategy(1000)`.
  
- `linsolve_cache::Union{Nothing, LinearSolve.LinearCache}`: Existing LinearSolve cache to reuse. If `nothing`, creates a new cache. Useful for iterative algorithms requiring factorization reuse.
  

**Type Parameters**
- `T<:Real`: The numeric type (e.g., Float64).
  
- `PrecisionMap<:Union{LinearMap{T}, AbstractMatrix{T}}`: The type of the precision matrix.
  

**Fields**
- `mean::Vector{T}`: The mean vector.
  
- `precision::PrecisionMap`: The precision matrix.
  
- `Q_sqrt::Union{Nothing, AbstractMatrix{T}}`: Square root of precision matrix for sampling.
  
- `linsolve_cache::LinearSolve.LinearCache`: The LinearSolve cache for efficient operations.
  
- `rbmc_strategy`: RBMC strategy for variance computation fallback.
  

**Notes**

The LinearSolve cache is constructed automatically (if not provided) and is used to compute means, variances,  samples, and other GMRF quantities efficiently. The algorithm choice determines which  optimization strategies (selected inversion, backward solve) are available. When selected inversion is not supported, marginal variances are computed using the configured RBMC strategy. Providing an existing `linsolve_cache` enables factorization reuse in iterative algorithms.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/gmrf.jl#L113-L143" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.precision_map' href='#GaussianMarkovRandomFields.precision_map'><span class="jlbinding">GaussianMarkovRandomFields.precision_map</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
precision_map(::AbstractGMRF)
```


Return the precision (inverse covariance) map of the GMRF.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/gmrf.jl#L71-L75" target="_blank" rel="noreferrer">source</a></Badge>



```julia
precision_map(d::ConstrainedGMRF)
```


Return the precision map of the constrained GMRF. Note: This is singular due to the constraints, but we return it for interface compliance. In practice, this should rarely be used directly due to singularity.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/arithmetic/constrained.jl#L132-L138" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.precision_matrix' href='#GaussianMarkovRandomFields.precision_matrix'><span class="jlbinding">GaussianMarkovRandomFields.precision_matrix</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
precision_matrix(::AbstractGMRF)
```


Return the precision (inverse covariance) matrix of the GMRF.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/gmrf.jl#L78-L82" target="_blank" rel="noreferrer">source</a></Badge>



```julia
precision_matrix(model::LatentModel; kwargs...)
```


Construct the precision matrix for the model given hyperparameter values.

**Arguments**
- `model`: The LatentModel instance
  
- `kwargs...`: Hyperparameter values as keyword arguments
  

**Returns**

A precision matrix (AbstractMatrix or LinearMap) for use in GMRF construction.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/latent_models/latent_model.jl#L66-L77" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.InformationVector' href='#GaussianMarkovRandomFields.InformationVector'><span class="jlbinding">GaussianMarkovRandomFields.InformationVector</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
InformationVector(data::AbstractVector)
```


Wrapper type for information vectors (Q * μ) used in GMRF construction. This allows distinguishing between constructors that take mean vectors  vs information vectors.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/gmrf.jl#L32-L38" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.information_vector' href='#GaussianMarkovRandomFields.information_vector'><span class="jlbinding">GaussianMarkovRandomFields.information_vector</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
information_vector(d::GMRF)
```


Return the information vector (Q * μ) for the GMRF. If stored, returns the cached value; otherwise computes it.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/gmrf.jl#L239-L244" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Metadata {#Metadata}
<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.MetaGMRF' href='#GaussianMarkovRandomFields.MetaGMRF'><span class="jlbinding">GaussianMarkovRandomFields.MetaGMRF</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
MetaGMRF{M <: GMRFMetadata, T, P, G <: AbstractGMRF{T, P}} <: AbstractGMRF{T, P}
```


A wrapper that combines a core GMRF with metadata of type M. This allows for specialized behavior based on the metadata type while preserving the computational efficiency of the underlying GMRF.

**Fields**
- `gmrf::G`: The core computational GMRF (parametric type)
  
- `metadata::M`: Domain-specific metadata
  

**Usage**

```julia
# Define metadata types
struct SpatialMetadata <: GMRFMetadata
    coordinates::Matrix{Float64}
    boundary_info::Vector{Int}
end

# Create wrapped GMRF
meta_gmrf = MetaGMRF(my_gmrf, SpatialMetadata(coords, boundary))

# Dispatch on metadata type for specialized behavior
function some_spatial_operation(mgmrf::MetaGMRF{SpatialMetadata})
    # Access coordinates via mgmrf.metadata.coordinates
    # Access GMRF via mgmrf.gmrf
end
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/metagmrf.jl#L14-L42" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.GMRFMetadata' href='#GaussianMarkovRandomFields.GMRFMetadata'><span class="jlbinding">GaussianMarkovRandomFields.GMRFMetadata</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
GMRFMetadata
```


Abstract base type for metadata that can be attached to GMRFs via MetaGMRF. Concrete subtypes should contain domain-specific information about the GMRF structure, coordinates, naming, etc.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/metagmrf.jl#L5-L11" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Arithmetic {#Arithmetic}
<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.condition_on_observations' href='#GaussianMarkovRandomFields.condition_on_observations'><span class="jlbinding">GaussianMarkovRandomFields.condition_on_observations</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



&quot;     condition_on_observations(         x::GMRF,         A::Union{AbstractMatrix,LinearMap},         Q_ϵ::Union{AbstractMatrix,LinearMap,Real},         y::AbstractVector=zeros(size(A)[1]),         b::AbstractVector=zeros(size(A)[1]);         # solver_blueprint parameter removed - no longer needed with LinearSolve     )

Condition a GMRF `x` on observations `y = A * x + b + ϵ` where `ϵ ~ N(0, Q_ϵ⁻¹)`.

**Arguments**
- `x::GMRF`: The GMRF to condition on.
  
- `A::Union{AbstractMatrix,LinearMap}`: The matrix `A`.
  
- `Q_ϵ::Union{AbstractMatrix,LinearMap, Real}`: The precision matrix of the        noise term `ϵ`. In case a real number is provided, it is interpreted        as a scalar multiple of the identity matrix.
  
- `y::AbstractVector=zeros(size(A)[1])`: The observations `y`; optional.
  
- `b::AbstractVector=zeros(size(A)[1])`: Offset vector `b`; optional.
  

**Keyword arguments**

**Note: solver_blueprint parameter removed - no longer needed with LinearSolve.jl**

**Returns**

A `GMRF` object representing the conditional GMRF `x | (y = A * x + b + ϵ)`.

**Notes**

This function is deprecated. Use `linear_condition`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/arithmetic/condition/linear.jl#L69-L99" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.linear_condition' href='#GaussianMarkovRandomFields.linear_condition'><span class="jlbinding">GaussianMarkovRandomFields.linear_condition</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
linear_condition(gmrf::GMRF; A, Q_ϵ, y, b=zeros(size(A, 1)))
```


Condition a GMRF on linear observations y = A * x + b + ϵ where ϵ ~ N(0, Q_ϵ^(-1)).

**Arguments**
- `gmrf::GMRF`: The prior GMRF
  
- `A::Union{AbstractMatrix, LinearMap}`: Observation matrix
  
- `Q_ϵ::Union{AbstractMatrix, LinearMap}`: Precision matrix of observation noise
  
- `y::AbstractVector`: Observation values
  
- `b::AbstractVector`: Offset vector (defaults to zeros)
  

**Returns**

A new `GMRF` representing the posterior distribution with updated mean and precision.

**Notes**

Uses information vector arithmetic for efficient conditioning without intermediate solves.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/arithmetic/condition/linear.jl#L25-L42" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.joint_gmrf' href='#GaussianMarkovRandomFields.joint_gmrf'><span class="jlbinding">GaussianMarkovRandomFields.joint_gmrf</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



&quot;     joint_gmrf(         x1::AbstractGMRF,         A::AbstractMatrix,         Q_ϵ::AbstractMatrix,         b::AbstractVector=spzeros(size(A)[1])     )

Return the joint GMRF of `x1` and `x2 = A * x1 + b + ϵ` where `ϵ ~ N(0, Q_ϵ⁻¹)`.

**Arguments**
- `x1::AbstractGMRF`: The first GMRF.
  
- `A::AbstractMatrix`: The matrix `A`.
  
- `Q_ϵ::AbstractMatrix`: The precision matrix of the noise term `ϵ`.
  
- `b::AbstractVector=spzeros(size(A)[1])`: Offset vector `b`; optional.
  

**Returns**

A `GMRF` object representing the joint GMRF of `x1` and `x2 = A * x1 + b + ϵ`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/arithmetic/joint.jl#L5-L24" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Spatiotemporal setting {#Spatiotemporal-setting}

### Types {#Types}
<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.AbstractSpatiotemporalGMRF' href='#GaussianMarkovRandomFields.AbstractSpatiotemporalGMRF'><span class="jlbinding">GaussianMarkovRandomFields.AbstractSpatiotemporalGMRF</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
AbstractSpatiotemporalGMRF
```


A spatiotemporal GMRF is a GMRF that explicitly encodes the spatial and temporal structure of the underlying random field. All time points are modelled in one joint GMRF. It provides utilities to get statistics, draw samples and get the spatial discretization at a given time.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/spdes/spatiotemporal/spatiotemporal_gmrf.jl#L20-L28" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.ImplicitEulerConstantMeshSTGMRF' href='#GaussianMarkovRandomFields.ImplicitEulerConstantMeshSTGMRF'><span class="jlbinding">GaussianMarkovRandomFields.ImplicitEulerConstantMeshSTGMRF</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
ImplicitEulerConstantMeshSTGMRF
```


A spatiotemporal GMRF with constant spatial discretization and an implicit Euler discretization of the temporal dynamics. Uses MetaGMRF for clean type structure.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/spdes/spatiotemporal/constant_mesh.jl#L36-L41" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.ConcreteConstantMeshSTGMRF' href='#GaussianMarkovRandomFields.ConcreteConstantMeshSTGMRF'><span class="jlbinding">GaussianMarkovRandomFields.ConcreteConstantMeshSTGMRF</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
ConcreteConstantMeshSTGMRF
```


A concrete implementation of a spatiotemporal GMRF with constant spatial discretization. Uses MetaGMRF for clean type structure.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/spdes/spatiotemporal/constant_mesh.jl#L44-L49" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Quantities {#Quantities}
<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.N_t' href='#GaussianMarkovRandomFields.N_t'><span class="jlbinding">GaussianMarkovRandomFields.N_t</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
N_t(::AbstractSpatiotemporalGMRF)
```


Return the number of time points in the spatiotemporal GMRF.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/spdes/spatiotemporal/spatiotemporal_gmrf.jl#L31-L35" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.time_means' href='#GaussianMarkovRandomFields.time_means'><span class="jlbinding">GaussianMarkovRandomFields.time_means</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
time_means(::AbstractSpatiotemporalGMRF)
```


Return the means of the spatiotemporal GMRF at each time point.

**Returns**
- A vector of means of length Nₜ, one for each time point.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/spdes/spatiotemporal/spatiotemporal_gmrf.jl#L38-L45" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.time_vars' href='#GaussianMarkovRandomFields.time_vars'><span class="jlbinding">GaussianMarkovRandomFields.time_vars</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
time_vars(::AbstractSpatiotemporalGMRF)
```


Return the marginal variances of the spatiotemporal GMRF at each time point.

**Returns**
- A vector of marginal variances of length Nₜ, one for each time point.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/spdes/spatiotemporal/spatiotemporal_gmrf.jl#L48-L55" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.time_stds' href='#GaussianMarkovRandomFields.time_stds'><span class="jlbinding">GaussianMarkovRandomFields.time_stds</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
time_stds(::AbstractSpatiotemporalGMRF)
```


Return the marginal standard deviations of the spatiotemporal GMRF at each time point.

**Returns**
- A vector of marginal standard deviations of length Nₜ, one for each time point.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/spdes/spatiotemporal/spatiotemporal_gmrf.jl#L58-L65" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.time_rands' href='#GaussianMarkovRandomFields.time_rands'><span class="jlbinding">GaussianMarkovRandomFields.time_rands</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
time_rands(::AbstractSpatiotemporalGMRF, rng::AbstractRNG)
```


Draw samples from the spatiotemporal GMRF at each time point.

**Returns**
- A vector of sample values of length Nₜ, one sample value vector for each time point.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/spdes/spatiotemporal/spatiotemporal_gmrf.jl#L68-L76" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.discretization_at_time' href='#GaussianMarkovRandomFields.discretization_at_time'><span class="jlbinding">GaussianMarkovRandomFields.discretization_at_time</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
discretization_at_time(::AbstractSpatiotemporalGMRF, t::Int)
```


Return the spatial discretization at time `t`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/spdes/spatiotemporal/spatiotemporal_gmrf.jl#L80-L84" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Utilities {#Utilities}
<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.spatial_to_spatiotemporal' href='#GaussianMarkovRandomFields.spatial_to_spatiotemporal'><span class="jlbinding">GaussianMarkovRandomFields.spatial_to_spatiotemporal</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
spatial_to_spatiotemporal(
    spatial_matrix::AbstractMatrix,
    t_idx::Int,
    N_t::Int,
)
```


Make a spatial matrix applicable to a spatiotemporal system at time index `t_idx`. Results in a matrix that selects the spatial information exactly at time `t_idx`.

**Arguments**
- `spatial_matrix::AbstractMatrix`: The spatial matrix.
  
- `t_idx::Integer`: The time index.
  
- `N_t::Integer`: The number of time points.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/spdes/spatiotemporal/utils.jl#L8-L23" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.kronecker_product_spatiotemporal_model' href='#GaussianMarkovRandomFields.kronecker_product_spatiotemporal_model'><span class="jlbinding">GaussianMarkovRandomFields.kronecker_product_spatiotemporal_model</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
kronecker_product_spatiotemporal_model(
    Q_t::AbstractMatrix,
    Q_s::AbstractMatrix,
    spatial_disc::FEMDiscretization;
    algorithm = nothing,
)
```


Create a spatiotemporal GMRF through a Kronecker product of the temporal and spatial precision matrices.

**Arguments**
- `Q_t::AbstractMatrix`: The temporal precision matrix.
  
- `Q_s::AbstractMatrix`: The spatial precision matrix.
  
- `spatial_disc::FEMDiscretization`: The spatial discretization.
  

**Keyword arguments**
- `algorithm`: The LinearSolve algorithm to use.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/spdes/spatiotemporal/product.jl#L5-L23" target="_blank" rel="noreferrer">source</a></Badge>

</details>

