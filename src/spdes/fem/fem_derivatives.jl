using Tensors, Ferrite, SparseArrays
using Tensors: ⊗

export derivative_matrices,
    second_derivative_matrices,
    geom_jacobian,
    shape_gradient_global,
    shape_gradient_local,
    shape_hessian_global,
    shape_hessian_local

"""
    shape_gradient_local(f::FEMDiscretization, shape_idx::Int, ξ)

Gradient of the shape function with index `shape_idx` with respect to the local
coordinates `ξ`.
"""
function shape_gradient_local(f::FEMDiscretization, shape_idx::Int, ξ)
    return Tensors.gradient(ξ -> Ferrite.value(f.interpolation, shape_idx, ξ), ξ)
end

"""
    shape_hessian_local(f::FEMDiscretization, shape_idx::Int, ξ)

Hessian of the shape function with index `shape_idx` with respect to the local
coordinates `ξ`.
"""
function shape_hessian_local(f::FEMDiscretization, shape_idx::Int, ξ)
    return Tensors.hessian(ξ -> Ferrite.value(f.interpolation, shape_idx, ξ), ξ)
end

"""
    geom_jacobian(f::FEMDiscretization, dof_coords, ξ)

Jacobian of the geometry mapping at the local coordinates `ξ` with node coordinates
`dof_coords`.
By "geometry mapping", we mean the mapping from the reference element to the
physical element.
"""
function geom_jacobian(f::FEMDiscretization, dof_coords, ξ)
    derivs = Ferrite.derivative(f.geom_interpolation, ξ)
    return sum([n ⊗ d for (n, d) ∈ zip(dof_coords, derivs)])
end

"""
    geom_hessian(f::FEMDiscretization, dof_coords, ξ)

Hessian of the geometry mapping at the local coordinates `ξ` with node coordinates
`dof_coords`.
By "geometry mapping", we mean the mapping from the reference element to the
physical element.
"""
function geom_hessian(f::FEMDiscretization, dof_coords, ξ)
    hessians = [
        Ferrite.hessian(ξ -> Ferrite.value(f.geom_interpolation, b, ξ), ξ) for
        b = 1:getnbasefunctions(f.geom_interpolation)
    ]
    return sum([n ⊗ h for (n, h) ∈ zip(dof_coords, hessians)])
end

"""
    shape_gradient_global(f::FEMDiscretization, dof_coords, shape_idx::Int, ξ; J⁻¹ = nothing)

Gradient of the shape function with index `shape_idx` in a cell with node coordinates
`dof_coords`, taken with respect to the global coordinates but computed in terms of the
local coordinates `ξ`.
"""
function shape_gradient_global(
    f::FEMDiscretization,
    dof_coords,
    shape_idx::Int,
    ξ;
    J⁻¹ = nothing,
)
    if J⁻¹ === nothing
        J⁻¹ = inv(geom_jacobian(f, dof_coords, ξ))
    end
    return J⁻¹' ⋅ shape_gradient_local(f, shape_idx, ξ)
end

"""
    shape_hessian_global(f::FEMDiscretization, dof_coords, shape_idx::Int, ξ; J⁻¹ = nothing, geo_hessian = nothing)

Hessian of the shape function with index `shape_idx` in a cell with node coordinates
`dof_coords`, taken with respect to the global coordinates but computed in terms of the
local coordinates `ξ`.
"""
function shape_hessian_global(
    f::FEMDiscretization,
    dof_coords,
    shape_idx::Int,
    ξ;
    J⁻¹ = nothing,
    geo_hessian = nothing,
)
    if J⁻¹ === nothing
        J⁻¹ = inv(geom_jacobian(f, dof_coords, ξ))
    end
    if geo_hessian === nothing
        geo_hessian = geom_hessian(f, dof_coords, ξ)
    end
    first_term = J⁻¹' ⋅ shape_hessian_local(f, shape_idx, ξ) ⋅ J⁻¹
    J⁻¹_hessian = -J⁻¹' ⋅ geo_hessian ⋅ J⁻¹'
    second_term = J⁻¹' ⋅ J⁻¹_hessian ⋅ shape_gradient_local(f, shape_idx, ξ)
    return first_term + second_term
end

"""
    derivative_matrices(f::FEMDiscretization{D}, X; derivative_idcs = [1])

Return a vector of matrices such that mats[k][i, j] is the derivative of the
j-th basis function at X[i], where the partial derivative index is given by
derivative_idcs[k].

# Examples
We're modelling a 2D function u(x, y) and we want the derivatives with respect
to y at two input points.

```@example
using Ferrite # hide
grid = generate_grid(Triangle, (20,20)) # hide
ip = Lagrange{2, RefTetrahedron, 1}() # hide
qr = QuadratureRule{2, RefTetrahedron}(2) # hide
disc = FEMDiscretization(grid, ip, qr)
X = [Tensors.Vec(0.11, 0.22), Tensors.Vec(-0.1, 0.4)]

mats = derivative_matrices(disc, X; derivative_idcs=[2])
```

`mats` contains a single matrix of size (2, ndofs(disc)) where the i-th row
contains the derivative of all basis functions with respect to y at X[i].
"""
function derivative_matrices(
    f::FEMDiscretization,
    X;
    derivative_idcs = [1],
    field = :default,
)
    if field == :default
        field = first(f.dof_handler.field_names)
    end
    dof_idcs = dof_range(f.dof_handler, field)
    peh = PointEvalHandler(f.grid, X)
    cc = CellCache(f.dof_handler)
    Is = [Int64[] for _ in derivative_idcs]
    Js = [Int64[] for _ in derivative_idcs]
    Vs = [Float64[] for _ in derivative_idcs]
    for i in eachindex(peh.cells)
        reinit!(cc, peh.cells[i])
        dof_coords = getcoordinates(cc)
        dofs = celldofs(cc)[dof_idcs]
        ξ = peh.local_coords[i]
        J⁻¹ = inv(geom_jacobian(f, dof_coords, ξ))
        get_grad = (b) -> shape_gradient_global(f, dof_coords, b, ξ; J⁻¹ = J⁻¹)
        derivatives = map(get_grad, 1:getnbasefunctions(f.interpolation))
        for (k, idx) in enumerate(derivative_idcs)
            append!(Is[k], repeat([i], length(dofs)))
            append!(Js[k], dofs)
            append!(Vs[k], map(x -> x[derivative_idcs[idx]], derivatives))
        end
    end
    mats = [
        sparse(Is[k], Js[k], Vs[k], length(X), ndofs(f)) for k = 1:length(derivative_idcs)
    ]
    return mats
end

"""
    second_derivative_matrices(f::FEMDiscretization{D}, X; derivative_idcs = [(1,1)])

Return a vector of matrices such that mats[k][i, j] is the second derivative of
the j-th basis function at X[i], where the partial derivative index is given by
derivative_idcs[k].
Note that the indices refer to the Hessian, i.e. (1, 2) corresponds to ∂²/∂x∂y.

# Examples
We're modelling a 2D function u(x, y) and we want to evaluate the Laplacian at
two input points.

```@example
using Ferrite # hide
grid = generate_grid(Triangle, (20,20)) # hide
ip = Lagrange{2, RefTetrahedron, 1}() # hide
qr = QuadratureRule{2, RefTetrahedron}(2) # hide
disc = FEMDiscretization(grid, ip, qr)
X = [Tensors.Vec(0.11, 0.22), Tensors.Vec(-0.1, 0.4)]

A, B = derivative_matrices(disc, X; derivative_idcs=[(1, 1), (2, 2)])
laplacian = A + B
```
"""
function second_derivative_matrices(
    f::FEMDiscretization,
    X;
    derivative_idcs = [(1, 1)],
    field = :default,
)
    if field == :default
        field = first(f.dof_handler.field_names)
    end
    dof_idcs = dof_range(f.dof_handler, field)
    peh = PointEvalHandler(f.grid, X)
    cc = CellCache(f.dof_handler)
    Is = [Int64[] for _ in derivative_idcs]
    Js = [Int64[] for _ in derivative_idcs]
    Vs = [Float64[] for _ in derivative_idcs]
    for i in eachindex(peh.cells)
        reinit!(cc, peh.cells[i])
        dof_coords = getcoordinates(cc)
        dofs = celldofs(cc)[dof_idcs]
        J⁻¹ = inv(geom_jacobian(f, dof_coords, peh.local_coords[i]))
        geo_hessian = geom_hessian(f, dof_coords, peh.local_coords[i])
        ξ = peh.local_coords[i]
        get_hessian =
            (b) -> shape_hessian_global(
                f,
                dof_coords,
                b,
                ξ;
                J⁻¹ = J⁻¹,
                geo_hessian = geo_hessian,
            )
        hessians = map(get_hessian, 1:getnbasefunctions(f.interpolation))
        for (k, idx) in enumerate(derivative_idcs)
            append!(Is[k], repeat([i], length(dofs)))
            append!(Js[k], dofs)
            append!(Vs[k], map(x -> x[idx...], hessians))
        end
    end
    mats = [
        sparse(Is[k], Js[k], Vs[k], length(X), ndofs(f)) for k = 1:length(derivative_idcs)
    ]
    return mats
end
