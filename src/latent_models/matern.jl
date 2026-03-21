using Distributions
using Ferrite
using GeometryBasics
using LinearSolve

export MaternModel

"""
    MaternModel{F<:FEMDiscretization,S<:Integer,Alg,C,P}(...)

A Matérn latent model for constructing spatial GMRFs from discretized Matérn SPDEs.

The MaternModel provides a structured way to define Matérn Gaussian Markov Random Fields
by discretizing the Matérn SPDE using finite element methods. The model stores the
discretization and smoothness parameter, while the range parameter is provided at
GMRF construction time.

# Mathematical Description

The Matérn SPDE is given by:
(κ² - Δ)^(α/2) u(x) = W(x), where α = ν + d/2

This leads to a Matérn covariance function with range and smoothness parameters.

# Two Construction Modes

1. **Direct construction**: Pass a pre-built `FEMDiscretization`
2. **Automatic construction**: Pass points, automatically creates convex hull mesh

# Hyperparameters
- `range`: Range parameter (range > 0) - controls spatial correlation distance

# Fields
- `discretization::F`: The finite element discretization
- `smoothness::S`: The smoothness parameter (Integer, controls differentiability)
- `alg::Alg`: LinearSolve algorithm for solving linear systems
- `constraint::C`: Optional constraint specification
- `observation_points::P`: N×D matrix of observation coordinates, or `nothing`

# Examples
```julia
# Direct construction
disc = FEMDiscretization(grid, interpolation, quadrature)
model = MaternModel(disc; smoothness = 2)
gmrf = model(range=2.0)

# Automatic construction from points (stores observation_points)
points = [0.0 0.0; 1.0 0.0; 0.5 1.0]  # N×2 matrix
model = MaternModel(points; smoothness = 1, element_order = 1)
gmrf = model(range=2.0)

# Convenience: evaluation matrix from stored points
A = evaluation_matrix(model)

# With custom algorithm
model = MaternModel(disc; smoothness = 2, alg = LDLtFactorization())
gmrf = model(range=2.0)
```
"""
struct MaternModel{F <: FEMDiscretization, S <: Integer, Alg, C, P} <: LatentModel
    discretization::F
    smoothness::S
    alg::Alg
    constraint::C
    observation_points::P

    function MaternModel{F, S, Alg, C, P}(discretization::F, smoothness::S, alg::Alg, constraint::C, observation_points::P) where {F <: FEMDiscretization, S <: Integer, Alg, C, P}
        smoothness >= 0 || throw(ArgumentError("Smoothness must be non-negative, got smoothness=$smoothness"))
        return new{F, S, Alg, C, P}(discretization, smoothness, alg, constraint, observation_points)
    end
end

"""
    MaternModel(discretization::F; smoothness::S, alg=CHOLMODFactorization(), constraint=nothing, observation_points=nothing) where {F<:FEMDiscretization, S<:Integer}

Direct construction with a pre-built FEMDiscretization.
"""
function MaternModel(discretization::F; smoothness::S, alg = CHOLMODFactorization(), constraint = nothing, observation_points = nothing) where {F <: FEMDiscretization, S <: Integer}
    n = ndofs(discretization)
    processed_constraint = _process_constraint(constraint, n)
    return MaternModel{F, S, typeof(alg), typeof(processed_constraint), typeof(observation_points)}(discretization, smoothness, alg, processed_constraint, observation_points)
end

"""
    MaternModel(points::AbstractMatrix; smoothness::Integer,
                element_order::Int = 1,
                interpolation = nothing,
                quadrature = nothing,
                alg = CHOLMODFactorization(),
                constraint = nothing)

Automatic construction: creates a convex hull around the 2D points and builds an appropriate
FEMDiscretization automatically.

# Arguments
- `points`: N×2 matrix where each row is a 2D point [x, y]
- `smoothness`: Smoothness parameter (Integer, ≥ 0) [REQUIRED KEYWORD]
- `element_order`: Order of finite element basis functions (default: 1)
- `interpolation`: Ferrite interpolation (default: Lagrange based on element_order)
- `quadrature`: Ferrite quadrature rule (default: based on element_order)
- `alg`: LinearSolve algorithm (default: CHOLMODFactorization())
- `constraint`: Optional constraint (default: nothing)
"""
function MaternModel(
        points::AbstractMatrix;
        smoothness::S,
        element_order::Int = 1,
        interpolation = nothing,
        quadrature = nothing,
        alg = CHOLMODFactorization(),
        constraint = nothing
    ) where {S <: Integer}

    # Input validation
    size(points, 2) == 2 || throw(ArgumentError("Points matrix must be N×2 for 2D points, got size $(size(points))"))
    size(points, 1) >= 3 || throw(ArgumentError("Need at least 3 points for convex hull, got $(size(points, 1))"))
    smoothness >= 0 || throw(ArgumentError("Smoothness must be non-negative, got smoothness=$smoothness"))
    element_order >= 1 || throw(ArgumentError("Element order must be >= 1, got element_order=$element_order"))

    # Convert matrix to vector of tuples for generate_mesh
    points_vec = [(points[i, 1], points[i, 2]) for i in 1:size(points, 1)]

    # Generate mesh using existing convex hull functionality
    grid = generate_mesh(
        points_vec;
        element_order = element_order
    )

    # Set up default interpolation and quadrature for 2D triangular elements
    if interpolation === nothing
        interpolation = element_order == 1 ? Lagrange{RefTriangle, 1}() : Lagrange{RefTriangle, element_order}()
    end

    if quadrature === nothing
        quadrature = QuadratureRule{RefTriangle}(2 * element_order)
    end

    # Create FEMDiscretization
    discretization = FEMDiscretization(grid, interpolation, quadrature)

    return MaternModel(discretization; smoothness = smoothness, alg = alg, constraint = constraint, observation_points = points)
end

function Base.length(model::MaternModel)
    return ndofs(model.discretization)
end

function hyperparameters(model::MaternModel)
    return (range = Real,)
end

function _validate_matern_parameters(; range::Real)
    range > 0 || throw(ArgumentError("Range parameter must be positive, got range=$range"))
    return nothing
end

function precision_matrix(model::MaternModel{F, S}; range::Real, kwargs...) where {F, S}
    _validate_matern_parameters(; range = range)

    # Extract dimension from discretization
    D = ndim(model.discretization)

    # Create MaternSPDE with the given range and model's smoothness
    spde = MaternSPDE{D}(range = range, smoothness = model.smoothness, σ² = 1.0)

    # Discretize using existing infrastructure - this returns a GMRF
    gmrf = discretize(spde, model.discretization)

    # Extract and return the precision matrix
    return precision_map(gmrf)
end

function mean(model::MaternModel; kwargs...)
    return zeros(length(model))
end

function constraints(model::MaternModel; kwargs...)
    return model.constraint
end

function model_name(::MaternModel)
    return :matern
end

"""
    evaluation_matrix(model::MaternModel)

Return the evaluation matrix mapping FEM DOFs to the stored observation points.
Only available when the model was constructed from points (i.e., `observation_points` is not `nothing`).
"""
function evaluation_matrix(model::MaternModel)
    model.observation_points === nothing && throw(
        ArgumentError(
            "No observation points stored. Use `evaluation_matrix(model, points)` " *
                "or construct the MaternModel from points to store them automatically."
        )
    )
    return evaluation_matrix(model.discretization, model.observation_points)
end

"""
    evaluation_matrix(model::MaternModel, points::AbstractMatrix)

Return the evaluation matrix mapping FEM DOFs to arbitrary spatial `points`.
Useful for prediction at new locations.

# Example
```julia
model = MaternModel(train_points; smoothness = 1)
A_pred = evaluation_matrix(model, test_points)
```
"""
function evaluation_matrix(model::MaternModel, points::AbstractMatrix)
    return evaluation_matrix(model.discretization, points)
end

"""
    PointEvaluationObsModel(model::MaternModel, family::Type{<:Distribution})

Convenience method that uses the observation points stored in a `MaternModel`.

Equivalent to `PointEvaluationObsModel(model.discretization, model.observation_points, family)`.

# Example
```julia
model = MaternModel(points; smoothness = 1)
obs_model = PointEvaluationObsModel(model, Normal)
```
"""
function PointEvaluationObsModel(model::MaternModel, family::Type{<:Distribution})
    model.observation_points === nothing && throw(
        ArgumentError(
            "No observation points stored in MaternModel. " *
                "Use `PointEvaluationObsModel(model, points, family)` instead, " *
                "or construct the MaternModel from points to store them automatically."
        )
    )
    return PointEvaluationObsModel(model.discretization, model.observation_points, family)
end

"""
    PointEvaluationObsModel(model::MaternModel, points::AbstractMatrix, family::Type{<:Distribution})

Create an observation model for function value observations at arbitrary spatial `points`.
Useful for prediction at new locations.

# Example
```julia
model = MaternModel(train_points; smoothness = 1)
obs_test = PointEvaluationObsModel(model, test_points, Normal)
```
"""
function PointEvaluationObsModel(model::MaternModel, points::AbstractMatrix, family::Type{<:Distribution})
    return PointEvaluationObsModel(model.discretization, points, family)
end
