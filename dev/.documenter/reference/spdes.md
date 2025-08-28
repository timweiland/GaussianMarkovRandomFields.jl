
# SPDEs {#SPDEs}
<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.SPDE' href='#GaussianMarkovRandomFields.SPDE'><span class="jlbinding">GaussianMarkovRandomFields.SPDE</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
SPDE
```


An abstract type for a stochastic partial differential equation (SPDE).


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/spdes/spde.jl#L3-L7" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.MaternSPDE' href='#GaussianMarkovRandomFields.MaternSPDE'><span class="jlbinding">GaussianMarkovRandomFields.MaternSPDE</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
MaternSPDE{D}(κ::Real, ν::Union{Integer, Rational}) where D
```


The Whittle-Matérn SPDE is given by

$$(κ^2 - Δ)^{\frac{α}{2}} u(x) = 𝒲(x), \quad \left( x \in \mathbb{R}^d,
α = ν + \frac{d}{2} \right),$$

where Δ is the Laplacian operator, $κ > 0$, $ν > 0$.

The stationary solutions to this SPDE are Matérn processes.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/spdes/matern.jl#L12-L25" target="_blank" rel="noreferrer">source</a></Badge>

</details>

<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.AdvectionDiffusionSPDE' href='#GaussianMarkovRandomFields.AdvectionDiffusionSPDE'><span class="jlbinding">GaussianMarkovRandomFields.AdvectionDiffusionSPDE</span></a> <Badge type="info" class="jlObjectType jlType" text="Type" /></summary>



```julia
AdvectionDiffusionSPDE{D}(κ::Real, α::Rational, H::AbstractMatrix,
γ::AbstractVector, c::Real, τ::Real) where {D}
```


Spatiotemporal advection-diffusion SPDE as proposed in [[2](/bibliography#Clarotto2024)]:

$$\left[ \frac{∂}{∂t} + \frac{1}{c} \left( κ^2 - ∇ ⋅ H ∇ \right)^\alpha
+ \frac{1}{c} γ ⋅ ∇ \right] X(t, s) = \frac{τ}{\sqrt{c}} Z(t, s),$$

where Z(t, s) is spatiotemporal noise which may be colored.


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/spdes/advection_diffusion.jl#L5-L17" target="_blank" rel="noreferrer">source</a></Badge>

</details>

