
# Autoregressive Models {#Autoregressive-Models}

For a hands-on example, check out the tutorial [Building autoregressive models](/tutorials/autoregressive_models#Building-autoregressive-models).
<details class='jldocstring custom-block' open>
<summary><a id='GaussianMarkovRandomFields.generate_car_model' href='#GaussianMarkovRandomFields.generate_car_model'><span class="jlbinding">GaussianMarkovRandomFields.generate_car_model</span></a> <Badge type="info" class="jlObjectType jlFunction" text="Function" /></summary>



```julia
generate_car_model(W::SparseMatrixCSC, ρ::Real; σ=1.0, μ=nothing)
```


Generate a conditional autoregressive model (CAR) in GMRF form from an adjacency matrix.

**Input**
- `W` – Adjacency / weight matrix. Specifies the conditional dependencies        between variables
  
- `ρ` – Weighting factor of the inter-node dependencies. Fulfills 0 &lt; ρ &lt; 1.
  
- `σ` – Variance scaling factor (i.e. output scale)
  
- `μ` – Mean vector
  

**Output**

A `GMRF` with the corresponding mean and precision.

**Algorithm**

The CAR is constructed using a variant of the graph Laplacian, i.e.

$$Q = \sigma^{-1} \cdot (W 1 - ρ \cdot W).$$


<Badge type="info" class="source-link" text="source"><a href="https://github.com/timweiland/GaussianMarkovRandomFields.jl/blob/dd37657e514bd21cd5478fa815e8a9868bc1d839/src/autoregressive/car.jl#L5-L30" target="_blank" rel="noreferrer">source</a></Badge>

</details>

