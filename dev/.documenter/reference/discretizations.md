
# Spatial and spatiotemporal discretizations {#Spatial-and-spatiotemporal-discretizations}

## Discretizing SPDEs {#Discretizing-SPDEs}
<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.discretize' href='#GaussianMarkovRandomFields.discretize'><span class="jlbinding">GaussianMarkovRandomFields.discretize</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
discretize(𝒟::MaternSPDE{D}, discretization::FEMDiscretization{D})::AbstractGMRF where {D}
```


Discretize a Matérn SPDE using a Finite Element Method (FEM) discretization. Computes the stiffness and (lumped) mass matrix, and then forms the precision matrix of the GMRF discretization.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/spdes/matern.jl#L186-L192" target="_blank" rel="noreferrer">source</a></Badge>



```julia
discretize(spde::AdvectionDiffusionSPDE, discretization::FEMDiscretization,
ts::AbstractVector{Float64}; colored_noise = false,
streamline_diffusion = false, h = 0.1) where {D}
```


Discretize an advection-diffusion SPDE using a constant spatial mesh. Streamline diffusion is an optional stabilization scheme for advection-dominated problems, which are known to be unstable. When using streamline diffusion, `h` may be passed to specify the mesh element size.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/spdes/advection_diffusion.jl#L100-L110" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Spatial discretization: FEM {#Spatial-discretization:-FEM}
<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.FEMDiscretization' href='#GaussianMarkovRandomFields.FEMDiscretization'><span class="jlbinding">GaussianMarkovRandomFields.FEMDiscretization</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
FEMDiscretization(
    grid::Ferrite.Grid,
    interpolation::Ferrite.Interpolation,
    quadrature_rule::Ferrite.QuadratureRule,
    fields = ((:u, nothing),),
    boundary_conditions = (),
)
```


A struct that contains all the information needed to discretize an (S)PDE using the Finite Element Method.

**Arguments**
- `grid::Ferrite.Grid`: The grid on which the discretization is defined.
  
- `interpolation::Ferrite.Interpolation`: The interpolation scheme, i.e. the                                         type of FEM elements.
  
- `quadrature_rule::Ferrite.QuadratureRule`: The quadrature rule.
  
- `fields::Vector{Tuple{Symbol, Union{Nothing, Ferrite.Interpolation}}}`:       The fields to be discretized. Each tuple contains the field name and       the geometric interpolation scheme. If the interpolation scheme is       `nothing`, `interpolation` is used for geometric interpolation.
  
- `boundary_conditions::Vector{Tuple{Ferrite.BoundaryCondition, Float64}}`:       The (soft) boundary conditions. Each tuple contains the boundary       condition and the noise standard deviation.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/spdes/fem/fem_discretization.jl#L6-L30" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.ndim' href='#GaussianMarkovRandomFields.ndim'><span class="jlbinding">GaussianMarkovRandomFields.ndim</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
ndim(f::FEMDiscretization)
```


Return the dimension of space in which the discretization is defined. Typically ndim(f) == 1, 2, or 3.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/spdes/fem/fem_discretization.jl#L120-L125" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='Ferrite.ndofs-Tuple{FEMDiscretization}' href='#Ferrite.ndofs-Tuple{FEMDiscretization}'><span class="jlbinding">Ferrite.ndofs</span></a> <Badge type="info" class="jlObjectType jlMethod" text="Method" /></summary>



```julia
ndofs(f::FEMDiscretization)
```


Return the number of degrees of freedom in the discretization.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/spdes/fem/fem_discretization.jl#L128-L132" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.evaluation_matrix' href='#GaussianMarkovRandomFields.evaluation_matrix'><span class="jlbinding">GaussianMarkovRandomFields.evaluation_matrix</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
evaluation_matrix(f::FEMDiscretization, X)
```


Return the matrix A such that A[i, j] is the value of the j-th basis function at the i-th point in X.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/spdes/fem/fem_discretization.jl#L135-L140" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.node_selection_matrix' href='#GaussianMarkovRandomFields.node_selection_matrix'><span class="jlbinding">GaussianMarkovRandomFields.node_selection_matrix</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
node_selection_matrix(f::FEMDiscretization, node_ids)
```


Return the matrix A such that A[i, j] = 1 if the j-th basis function is associated with the i-th node in node_ids.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/spdes/fem/fem_discretization.jl#L165-L170" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.derivative_matrices' href='#GaussianMarkovRandomFields.derivative_matrices'><span class="jlbinding">GaussianMarkovRandomFields.derivative_matrices</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
derivative_matrices(f::FEMDiscretization{D}, X; derivative_idcs = [1])
```


Return a vector of matrices such that mats[k][i, j] is the derivative of the j-th basis function at X[i], where the partial derivative index is given by derivative_idcs[k].

**Examples**

We&#39;re modelling a 2D function u(x, y) and we want the derivatives with respect to y at two input points.

```julia
using Ferrite # hide
grid = generate_grid(Triangle, (20,20)) # hide
ip = Lagrange{2, RefTetrahedron, 1}() # hide
qr = QuadratureRule{2, RefTetrahedron}(2) # hide
disc = FEMDiscretization(grid, ip, qr)
X = [Tensors.Vec(0.11, 0.22), Tensors.Vec(-0.1, 0.4)]

mats = derivative_matrices(disc, X; derivative_idcs=[2])
```


`mats` contains a single matrix of size (2, ndofs(disc)) where the i-th row contains the derivative of all basis functions with respect to y at X[i].


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/spdes/fem/fem_derivatives.jl#L111-L135" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.second_derivative_matrices' href='#GaussianMarkovRandomFields.second_derivative_matrices'><span class="jlbinding">GaussianMarkovRandomFields.second_derivative_matrices</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
second_derivative_matrices(f::FEMDiscretization{D}, X; derivative_idcs = [(1,1)])
```


Return a vector of matrices such that mats[k][i, j] is the second derivative of the j-th basis function at X[i], where the partial derivative index is given by derivative_idcs[k]. Note that the indices refer to the Hessian, i.e. (1, 2) corresponds to ∂²/∂x∂y.

**Examples**

We&#39;re modelling a 2D function u(x, y) and we want to evaluate the Laplacian at two input points.

```julia
using Ferrite # hide
grid = generate_grid(Triangle, (20,20)) # hide
ip = Lagrange{2, RefTetrahedron, 1}() # hide
qr = QuadratureRule{2, RefTetrahedron}(2) # hide
disc = FEMDiscretization(grid, ip, qr)
X = [Tensors.Vec(0.11, 0.22), Tensors.Vec(-0.1, 0.4)]

A, B = derivative_matrices(disc, X; derivative_idcs=[(1, 1), (2, 2)])
laplacian = A + B
```



<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/spdes/fem/fem_derivatives.jl#L170-L193" target="_blank" rel="noreferrer">source</a></Badge>

</details>


### Utilities {#Utilities}
<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.assemble_mass_matrix' href='#GaussianMarkovRandomFields.assemble_mass_matrix'><span class="jlbinding">GaussianMarkovRandomFields.assemble_mass_matrix</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
assemble_mass_matrix(
    Ce::SparseMatrixCSC,
    cellvalues::CellValues,
    interpolation;
    lumping = true,
)
```


Assemble the mass matrix `Ce` for the given cell values.

**Arguments**
- `Ce::SparseMatrixCSC`: The mass matrix.
  
- `cellvalues::CellValues`: Ferrite cell values.
  
- `interpolation::Interpolation`: The interpolation scheme.
  
- `lumping::Bool=true`: Whether to lump the mass matrix.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/spdes/fem/utils.jl#L35-L50" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.assemble_diffusion_matrix' href='#GaussianMarkovRandomFields.assemble_diffusion_matrix'><span class="jlbinding">GaussianMarkovRandomFields.assemble_diffusion_matrix</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
assemble_diffusion_matrix(
    Ge::SparseMatrixCSC,
    cellvalues::CellValues;
    diffusion_factor = I,
)
```


Assemble the diffusion matrix `Ge` for the given cell values.

**Arguments**
- `Ge::SparseMatrixCSC`: The diffusion matrix.
  
- `cellvalues::CellValues`: Ferrite cell values.
  
- `diffusion_factor=I`: The diffusion factor.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/spdes/fem/utils.jl#L81-L94" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.assemble_advection_matrix' href='#GaussianMarkovRandomFields.assemble_advection_matrix'><span class="jlbinding">GaussianMarkovRandomFields.assemble_advection_matrix</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
assemble_advection_matrix(
    Be::SparseMatrixCSC,
    cellvalues::CellValues;
    advection_velocity = 1,
)
```


Assemble the advection matrix `Be` for the given cell values.

**Arguments**
- `Be::SparseMatrixCSC`: The advection matrix.
  
- `cellvalues::CellValues`: Ferrite cell values.
  
- `advection_velocity=1`: The advection velocity.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/spdes/fem/utils.jl#L121-L134" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.lump_matrix' href='#GaussianMarkovRandomFields.lump_matrix'><span class="jlbinding">GaussianMarkovRandomFields.lump_matrix</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
lump_matrix(A::AbstractMatrix, ::Lagrange{D, S, 1}) where {D, S}
```


Lump a matrix by summing over the rows.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/spdes/fem/utils.jl#L10-L14" target="_blank" rel="noreferrer">source</a></Badge>



```julia
lump_matrix(A::AbstractMatrix, ::Lagrange)
```


Lump a matrix through HRZ lumping. Fallback for non-linear elements. Row-summing cannot be used for non-linear elements, because it does not ensure positive definiteness.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/spdes/fem/utils.jl#L19-L26" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.assemble_streamline_diffusion_matrix' href='#GaussianMarkovRandomFields.assemble_streamline_diffusion_matrix'><span class="jlbinding">GaussianMarkovRandomFields.assemble_streamline_diffusion_matrix</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
assemble_streamline_diffusion_matrix(
    Ge::SparseMatrixCSC,
    cellvalues::CellValues,
    advection_velocity,
    h,
)
```


Assemble the streamline diffusion matrix `Ge` for the given cell values.

**Arguments**
- `Ge::SparseMatrixCSC`: The streamline diffusion matrix.
  
- `cellvalues::CellValues`: Ferrite cell values.
  
- `advection_velocity`: The advection velocity.
  
- `h`: The mesh size.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/spdes/fem/utils.jl#L161-L176" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.apply_soft_constraints!' href='#GaussianMarkovRandomFields.apply_soft_constraints!'><span class="jlbinding">GaussianMarkovRandomFields.apply_soft_constraints!</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
apply_soft_constraints!(K, f_rhs, ch, constraint_noise; Q_rhs = nothing, Q_rhs_sqrt = nothing)
```


Apply soft constraints to the Gaussian relation

$$\mathbf{K} \mathbf{u} \sim \mathcal{N}(\mathbf{f}_{\text{rhs}}, \mathbf{Q}_{\text{rhs}}^{-1})$$

Soft means that the constraints are fulfilled up to noise of magnitude specified by `constraint_noise`.

Modifies `K` and `f_rhs` in place. If `Q_rhs` and `Q_rhs_sqrt` are provided, they are modified in place as well.

**Arguments**
- `K::SparseMatrixCSC`: Stiffness matrix.
  
- `f_rhs::AbstractVector`: Right-hand side.
  
- `ch::ConstraintHandler`: Constraint handler.
  
- `constraint_noise::Vector{Float64}`: Noise for each constraint.
  
- `Q_rhs::Union{Nothing, SparseMatrixCSC}`: Covariance matrix for the right-hand                                           side.
  
- `Q_rhs_sqrt::Union{Nothing, SparseMatrixCSC}`: Square root of the covariance                                                matrix for the right-hand side.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/spdes/fem/utils.jl#L205-L228" target="_blank" rel="noreferrer">source</a></Badge>

</details>


## Temporal discretization and state-space models {#Temporal-discretization-and-state-space-models}
<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.JointSSMMatrices' href='#GaussianMarkovRandomFields.JointSSMMatrices'><span class="jlbinding">GaussianMarkovRandomFields.JointSSMMatrices</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
JointSSMMatrices
```


Abstract type for the matrices defining the transition of a certain linear state-space model of the form

$$G(Δt) x_{k+1} ∣ xₖ ∼ 𝒩(M(Δt) xₖ, Σ)$$

**Fields**
- `Δt::Real`: Time step.
  
- `G::LinearMap`: Transition matrix.
  
- `M::LinearMap`: Observation matrix.
  
- `Σ⁻¹::LinearMap`: Transition precision map.
  
- `Σ⁻¹_sqrt::LinearMap`: Square root of the transition precision map.
  
- `constraint_handler`: Ferrite constraint handler.
  
- `constraint_noise`: Constraint noise.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/spdes/spatiotemporal/ssm/linear_ssm.jl#L6-L24" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.joint_ssm' href='#GaussianMarkovRandomFields.joint_ssm'><span class="jlbinding">GaussianMarkovRandomFields.joint_ssm</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
joint_ssm(x₀::GMRF, ssm_matrices::Function, ts::AbstractVector)
```


Form the joint GMRF for the linear state-space model given by

$$G(Δtₖ) x_{k+1} ∣ xₖ ∼ 𝒩(M(Δtₖ) xₖ, Σ)$$

at time points given by `ts` (from which the Δtₖ are computed).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/spdes/spatiotemporal/ssm/linear_ssm.jl#L36-L46" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.ImplicitEulerSSM' href='#GaussianMarkovRandomFields.ImplicitEulerSSM'><span class="jlbinding">GaussianMarkovRandomFields.ImplicitEulerSSM</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
ImplicitEulerSSM{X,S,GF,MF,MIF,BF,BIF,TS,C,V}(
    x₀::X,
    G::GF,
    M::MF,
    M⁻¹::MIF,
    β::BF,
    β⁻¹::BIF,
    spatial_noise::S,
    ts::TS,
    constraint_handler::C,
    constraint_noise::V,
)
```


State-space model for the implicit Euler discretization of a stochastic differential equation.

The state-space model is given by

```
G(Δt) xₖ₊₁ = M(Δt) xₖ + M(Δt) β(Δt) zₛ
```


where `zₛ` is (possibly colored) spatial noise. 


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/spdes/spatiotemporal/ssm/implicit_euler_ssm.jl#L6-L30" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.ImplicitEulerJointSSMMatrices' href='#GaussianMarkovRandomFields.ImplicitEulerJointSSMMatrices'><span class="jlbinding">GaussianMarkovRandomFields.ImplicitEulerJointSSMMatrices</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
ImplicitEulerJointSSMMatrices{T,GM,MM,SM,SQRT,C,V}(
    ssm::ImplicitEulerSSM,
    Δt::Real
)
```


Construct the joint state-space model matrices for the implicit Euler discretization scheme.

**Arguments**
- `ssm::ImplicitEulerSSM`: The implicit Euler state-space model.
  
- `Δt::Real`: The time step.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/spdes/spatiotemporal/ssm/implicit_euler_ssm.jl#L70-L82" target="_blank" rel="noreferrer">source</a></Badge>

</details>

