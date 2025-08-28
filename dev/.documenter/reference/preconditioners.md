
# Preconditioners {#Preconditioners}
<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.AbstractPreconditioner' href='#GaussianMarkovRandomFields.AbstractPreconditioner'><span class="jlbinding">GaussianMarkovRandomFields.AbstractPreconditioner</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
AbstractPreconditioner
```


Abstract type for preconditioners. Should implement the following methods:
- ldiv!(y, P::AbstractPreconditioner, x::AbstractVector)
  
- ldiv!(P::AbstractPreconditioner, x::AbstractVector)
  
- \(P::AbstractPreconditioner, x::AbstractVector)
  
- size(P::AbstractPreconditioner)
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/preconditioners/preconditioner.jl#L6-L15" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.BlockJacobiPreconditioner' href='#GaussianMarkovRandomFields.BlockJacobiPreconditioner'><span class="jlbinding">GaussianMarkovRandomFields.BlockJacobiPreconditioner</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
BlockJacobiPreconditioner
```


A preconditioner that uses a block Jacobi preconditioner, i.e. P = diag(A₁, A₂, ...), where each Aᵢ is a preconditioner for a block of the matrix.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/preconditioners/block_jacobi.jl#L6-L11" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.FullCholeskyPreconditioner' href='#GaussianMarkovRandomFields.FullCholeskyPreconditioner'><span class="jlbinding">GaussianMarkovRandomFields.FullCholeskyPreconditioner</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
FullCholeskyPreconditioner
```


A preconditioner that uses a full Cholesky factorization of the matrix, i.e. P = A, so P⁻¹ = A⁻¹. Does not make sense to use on its own, but can be used as a building block for more complex preconditioners.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/preconditioners/full_cholesky.jl#L7-L14" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.temporal_block_gauss_seidel' href='#GaussianMarkovRandomFields.temporal_block_gauss_seidel'><span class="jlbinding">GaussianMarkovRandomFields.temporal_block_gauss_seidel</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
temporal_block_gauss_seidel(A, block_size)
```


Construct a temporal block Gauss-Seidel preconditioner for a spatiotemporal matrix with constant spatial mesh size (and thus constant spatial block size).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/preconditioners/spatiotemporal_preconditioner.jl#L5-L10" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.TridiagonalBlockGaussSeidelPreconditioner' href='#GaussianMarkovRandomFields.TridiagonalBlockGaussSeidelPreconditioner'><span class="jlbinding">GaussianMarkovRandomFields.TridiagonalBlockGaussSeidelPreconditioner</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
TridiagonalBlockGaussSeidelPreconditioner{T}(D_blocks, L_blocks)
TridiagonalBlockGaussSeidelPreconditioner{T}(D⁻¹_blocks, L_blocks)
```


Block Gauss-Seidel preconditioner for block tridiagonal matrices. For a matrix given by

$$A = \begin{bmatrix}
D₁ & L₁ᵀ & 0 & \cdots & 0 \\
L₁ & D₂ & L₂ᵀ & 0 & \cdots \\
0 & L₂ & D₃ & L₃ᵀ & \cdots \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & \cdots & 0 & Lₙ₋₁ & Lₙ
\end{bmatrix}$$

this preconditioner is given by

$$P = \begin{bmatrix}
D₁ & 0 & \cdots & 0 \\
L₁ & D₂ & 0 & \cdots \\
0 & L₂ & D₃ & \cdots \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & \cdots & 0 & Lₙ₋₁ & Dₙ
\end{bmatrix}$$

Solving linear systems with the preconditioner is made efficient through block forward / backward substitution. The diagonal blocks must be inverted. As such, they may be specified
1. directly as matrices: in this case they will be transformed into `FullCholeskyPreconditioner`s.
  
2. in terms of their invertible preconditioners
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/preconditioners/tridiag_block_gauss_seidel.jl#L7-L42" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.TridiagSymmetricBlockGaussSeidelPreconditioner' href='#GaussianMarkovRandomFields.TridiagSymmetricBlockGaussSeidelPreconditioner'><span class="jlbinding">GaussianMarkovRandomFields.TridiagSymmetricBlockGaussSeidelPreconditioner</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
TridiagSymmetricBlockGaussSeidelPreconditioner{T}(D_blocks, L_blocks)
TridiagSymmetricBlockGaussSeidelPreconditioner{T}(D⁻¹_blocks, L_blocks)
```


Symmetric Block Gauss-Seidel preconditioner for symmetric block tridiagonal matrices. For a symmetric matrix given by the block decomposition A = L + D + Lᵀ, where L is strictly lower triangular and D is diagonal, this preconditioner is given by P = (L + D) D⁻¹ (L + D)ᵀ ≈ A.

Solving linear systems with the preconditioner is made efficient through block forward / backward substitution. The diagonal blocks must be inverted. As such, they may be specified
1. directly as matrices: in this case they will be transformed into `FullCholeskyPreconditioner`s.
  
2. in terms of their invertible preconditioners
  


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/preconditioners/tridiag_block_gauss_seidel.jl#L103-L121" target="_blank" rel="noreferrer">source</a></Badge>

</details>

