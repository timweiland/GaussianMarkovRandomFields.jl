# Linear Maps

The construction of GMRFs involves various kinds of structured matrices.
These structures may be leveraged in downstream computations to save compute
and memory.
But to make this possible, we need to actually keep track of these structures - 
which we achieve through diverse subtypes of
[LinearMap](https://julialinearalgebra.github.io/LinearMaps.jl/stable/).

```@docs
SymmetricBlockTridiagonalMap
SSMBidiagonalMap
OuterProductMap
LinearMapWithSqrt
CholeskySqrt
ZeroMap
ADJacobianMap
ADJacobianAdjointMap
```
