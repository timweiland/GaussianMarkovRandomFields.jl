
# Linear Maps {#Linear-Maps}

The construction of GMRFs involves various kinds of structured matrices. These structures may be leveraged in downstream computations to save compute and memory. But to make this possible, we need to actually keep track of these structures -  which we achieve through diverse subtypes of [LinearMap](https://julialinearalgebra.github.io/LinearMaps.jl/stable/).
<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.SymmetricBlockTridiagonalMap' href='#GaussianMarkovRandomFields.SymmetricBlockTridiagonalMap'><span class="jlbinding">GaussianMarkovRandomFields.SymmetricBlockTridiagonalMap</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
SymmetricBlockTridiagonalMap(
    diagonal_blocks::Tuple{LinearMap{T},Vararg{LinearMap{T},ND}},
    off_diagonal_blocks::Tuple{LinearMap{T},Vararg{LinearMap{T},NOD}},
)
```


A linear map representing a symmetric block tridiagonal matrix with diagonal blocks `diagonal_blocks` and lower off-diagonal blocks `off_diagonal_blocks`.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/linear_maps/symmetric_block_tridiagonal.jl#L9-L18" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.SSMBidiagonalMap' href='#GaussianMarkovRandomFields.SSMBidiagonalMap'><span class="jlbinding">GaussianMarkovRandomFields.SSMBidiagonalMap</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
SSMBidiagonalMap{T}(
    A::LinearMap{T},
    B::LinearMap{T},
    C::LinearMap{T},
    N_t::Int,
)
```


Represents the block-bidiagonal map given by the (N_t) x (N_t - 1) sized block structure:

$$\begin{bmatrix}
A & 0 & \cdots & 0 \\
B & C & \cdots & 0 \\
0 & B & C & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & B
\end{bmatrix}$$

which occurs as a square root in the discretization of GMRF-based state-space models. `N_t` is the total number of blocks along the rows.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/linear_maps/ssm_bidiagonal.jl#L6-L29" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.OuterProductMap' href='#GaussianMarkovRandomFields.OuterProductMap'><span class="jlbinding">GaussianMarkovRandomFields.OuterProductMap</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
OuterProductMap{T}(
    A::LinearMap{T},
    Q::LinearMap{T},
)
```


Represents the outer product A&#39; Q A, without actually forming it in memory.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/linear_maps/outer_product.jl#L8-L15" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.LinearMapWithSqrt' href='#GaussianMarkovRandomFields.LinearMapWithSqrt'><span class="jlbinding">GaussianMarkovRandomFields.LinearMapWithSqrt</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
LinearMapWithSqrt{T}(
    A::LinearMap{T},
    A_sqrt::LinearMap{T},
)
```


A symmetric positive definite linear map `A` with known square root `A_sqrt`, i.e. `A = A_sqrt * A_sqrt'`. Behaves just like `A`, but taking the square root directly returns `A_sqrt`.

**Arguments**
- `A::LinearMap{T}`: The linear map `A`.
  
- `A_sqrt::LinearMap{T}`: The square root of `A`.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/linear_maps/linear_map_with_sqrt.jl#L6-L19" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.CholeskySqrt' href='#GaussianMarkovRandomFields.CholeskySqrt'><span class="jlbinding">GaussianMarkovRandomFields.CholeskySqrt</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
CholeskySqrt(cho::Union{Cholesky{T},SparseArrays.CHOLMOD.Factor{T}})
```


A linear map representing the square root obtained from a Cholesky factorization, i.e. for `A = L * L'`, this map represents `L`.

**Arguments**
- `cho::Union{Cholesky{T},SparseArrays.CHOLMOD.Factor{T}}`:   The Cholesky factorization of a symmetric positive definite matrix.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/linear_maps/cholesky_sqrt.jl#L17-L26" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.CholeskyFactorizedMap' href='#GaussianMarkovRandomFields.CholeskyFactorizedMap'><span class="jlbinding">GaussianMarkovRandomFields.CholeskyFactorizedMap</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
CholeskyFactorizedMap{T,C}(cho::C) where {T,C}
```


A linear map represented in terms of its Cholesky factorization, i.e. for `A = L * L'`, this map represents `A`.

**Type Parameters**
- `T`: Element type of the matrix
  
- `C`: Type of the Cholesky factorization
  

**Arguments**
- `cho`: The Cholesky factorization of a symmetric positive definite matrix. Can be `Cholesky`, `SparseArrays.CHOLMOD.Factor`, or `LDLFactorization`.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/linear_maps/cholesky_factorized_map.jl#L9-L22" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.ZeroMap' href='#GaussianMarkovRandomFields.ZeroMap'><span class="jlbinding">GaussianMarkovRandomFields.ZeroMap</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
ZeroMap{T}(N::Int, M::Int)
```


A linear map that maps all vectors to the zero vector.

**Arguments**
- `N::Int`: Output dimension
  
- `M::Int`: Input dimension
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/linear_maps/zero.jl#L5-L13" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.ADJacobianMap' href='#GaussianMarkovRandomFields.ADJacobianMap'><span class="jlbinding">GaussianMarkovRandomFields.ADJacobianMap</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
ADJacobianMap(f::Function, x₀::AbstractVector{T}, N_outputs::Int)
```


A linear map representing the Jacobian of `f` at `x₀`. Uses forward-mode AD in a matrix-free way, i.e. we do not actually store the Jacobian in memory and only compute JVPs.

Requires ForwardDiff.jl!

**Arguments**
- `f::Function`: Function to differentiate.
  
- `x₀::AbstractVector{T}`: Input vector at which to evaluate the Jacobian.
  
- `N_outputs::Int`: Output dimension of `f`.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/linear_maps/ad_jacobian.jl#L5-L18" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.ADJacobianAdjointMap' href='#GaussianMarkovRandomFields.ADJacobianAdjointMap'><span class="jlbinding">GaussianMarkovRandomFields.ADJacobianAdjointMap</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
ADJacobianAdjointMap{T}(f::Function, x₀::AbstractVector{T}, N_outputs::Int)
```


A linear map representing the adjoint of the Jacobian of `f` at `x₀`. Uses reverse-mode AD in a matrix-free way, i.e. we do not actually store the Jacobian in memory and only compute VJPs.

Requires Zygote.jl!

**Arguments**
- `f::Function`: Function to differentiate.
  
- `x₀::AbstractVector{T}`: Input vector at which to evaluate the Jacobian.
  
- `N_outputs::Int`: Output dimension of `f`.
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/linear_maps/ad_jacobian.jl#L37-L50" target="_blank" rel="noreferrer">source</a></Badge>

</details>

