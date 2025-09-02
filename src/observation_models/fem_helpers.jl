using Distributions

export PointEvaluationObsModel, PointDerivativeObsModel, PointSecondDerivativeObsModel

"""
    PointEvaluationObsModel(fem::FEMDiscretization, points::AbstractMatrix, family::Type{<:Distribution}) -> LinearlyTransformedObservationModel

Create an observation model for observing function values at specified points using a FEM discretization.

This helper automatically constructs a `LinearlyTransformedObservationModel` where:
- The design matrix comes from `evaluation_matrix(fem, points)`
- The base observation model is `ExponentialFamily(family)`

# Arguments
- `fem`: FEM discretization defining the basis functions
- `points`: N×D matrix where each row is a spatial point to observe
- `family`: Distribution family for observations (e.g., `Normal`, `Poisson`)

# Returns
`LinearlyTransformedObservationModel` ready for materialization with data

# Example
```julia
# Create FEM discretization
grid = generate_grid(Triangle, (20, 20))
ip = Lagrange{RefTriangle, 1}()
qr = QuadratureRule{RefTriangle}(2)
fem = FEMDiscretization(grid, ip, qr)

# Observation points
points = [0.1 0.2; 0.5 0.7; 0.9 0.3]  # 3 points in 2D

# Create observation model for Poisson count data
obs_model = PointEvaluationObsModel(fem, points, Poisson)

# Materialize with data
y = [2, 5, 1]  # Count observations
obs_lik = obs_model(y)  # Ready for loglik evaluation
```
"""
function PointEvaluationObsModel(fem::FEMDiscretization, points::AbstractMatrix, family::Type{<:Distribution})
    design_matrix = evaluation_matrix(fem, points)
    base_model = ExponentialFamily(family)
    return LinearlyTransformedObservationModel(base_model, design_matrix)
end

"""
    PointDerivativeObsModel(fem::FEMDiscretization, points::AbstractMatrix, family::Type{<:Distribution}; derivative_idcs=nothing) -> LinearlyTransformedObservationModel

Create an observation model for observing first derivatives at specified points using a FEM discretization.

This helper automatically constructs a `LinearlyTransformedObservationModel` where:
- The design matrix comes from `derivative_matrices(fem, points; derivative_idcs)`
- The base observation model is `ExponentialFamily(family)`

# Arguments
- `fem`: FEM discretization defining the basis functions
- `points`: N×D matrix where each row is a spatial point to observe derivatives
- `family`: Distribution family for observations (e.g., `Normal`, `Poisson`)

# Keywords
- `derivative_idcs`: Indices of partial derivatives to compute. Defaults to `[1, 2, ..., ndim(fem)]` (all spatial directions)

# Returns
`LinearlyTransformedObservationModel` ready for materialization with data

# Example
```julia
# Create observation model for x-derivatives of a 2D function
obs_model = PointDerivativeObsModel(fem, points, Normal; derivative_idcs=[1])

# Or observe both x and y derivatives
obs_model = PointDerivativeObsModel(fem, points, Normal; derivative_idcs=[1, 2])
# This creates concatenated observations: [∂u/∂x at all points, ∂u/∂y at all points]

# Materialize with gradient data
y = [0.1, 0.3, -0.2, 0.5, 0.1, 0.0]  # [∂u/∂x₁, ∂u/∂x₂, ∂u/∂x₃, ∂u/∂y₁, ∂u/∂y₂, ∂u/∂y₃]
obs_lik = obs_model(y; σ=0.1)
```
"""
function PointDerivativeObsModel(fem::FEMDiscretization, points::AbstractMatrix, family::Type{<:Distribution}; derivative_idcs = nothing)
    if derivative_idcs === nothing
        derivative_idcs = collect(1:ndim(fem))
    end

    derivative_mats = derivative_matrices(fem, points; derivative_idcs = derivative_idcs)

    # Concatenate matrices vertically to create single design matrix
    if length(derivative_mats) == 1
        design_matrix = derivative_mats[1]
    else
        design_matrix = vcat(derivative_mats...)
    end

    base_model = ExponentialFamily(family)
    return LinearlyTransformedObservationModel(base_model, design_matrix)
end

"""
    PointSecondDerivativeObsModel(fem::FEMDiscretization, points::AbstractMatrix, family::Type{<:Distribution}; derivative_idcs=nothing) -> LinearlyTransformedObservationModel

Create an observation model for observing second derivatives at specified points using a FEM discretization.

This helper automatically constructs a `LinearlyTransformedObservationModel` where:
- The design matrix comes from `second_derivative_matrices(fem, points; derivative_idcs)`
- The base observation model is `ExponentialFamily(family)`

# Arguments
- `fem`: FEM discretization defining the basis functions
- `points`: N×D matrix where each row is a spatial point to observe second derivatives
- `family`: Distribution family for observations (e.g., `Normal`, `Poisson`)

# Keywords
- `derivative_idcs`: Indices of second partial derivatives to compute. Defaults to diagonal terms `[(1,1), (2,2), ...]` up to `ndim(fem)`

# Returns
`LinearlyTransformedObservationModel` ready for materialization with data

# Example
```julia
# Create observation model for Laplacian (∂²u/∂x² + ∂²u/∂y²) observations
obs_model = PointSecondDerivativeObsModel(fem, points, Normal; derivative_idcs=[(1,1), (2,2)])

# Materialize with Laplacian data
y = [0.05, -0.1, 0.2, 0.15, -0.05, 0.1]  # [∂²u/∂x²₁, ∂²u/∂x²₂, ∂²u/∂x²₃, ∂²u/∂y²₁, ∂²u/∂y²₂, ∂²u/∂y²₃]
obs_lik = obs_model(y; σ=0.01)
```
"""
function PointSecondDerivativeObsModel(fem::FEMDiscretization, points::AbstractMatrix, family::Type{<:Distribution}; derivative_idcs = nothing)
    if derivative_idcs === nothing
        D = ndim(fem)
        derivative_idcs = [(i, i) for i in 1:D]  # Diagonal terms
    end

    second_derivative_mats = second_derivative_matrices(fem, points; derivative_idcs = derivative_idcs)

    # Concatenate matrices vertically to create single design matrix
    if length(second_derivative_mats) == 1
        design_matrix = second_derivative_mats[1]
    else
        design_matrix = vcat(second_derivative_mats...)
    end

    base_model = ExponentialFamily(family)
    return LinearlyTransformedObservationModel(base_model, design_matrix)
end
