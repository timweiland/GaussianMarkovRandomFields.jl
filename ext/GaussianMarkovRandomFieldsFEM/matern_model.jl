"""
    MaternModel(discretization::F; smoothness::S, alg=CHOLMODFactorization(), constraint=nothing, observation_points=nothing) where {F<:FEMDiscretization, S<:Integer}

Direct construction of a Matérn latent model with a pre-built FEMDiscretization.

The MaternModel provides a structured way to define Matérn Gaussian Markov
Random Fields by discretizing the Matérn SPDE using finite element methods.

# Examples
```julia
disc = FEMDiscretization(grid, interpolation, quadrature)
model = MaternModel(disc; smoothness = 2)
gmrf = model(τ=1.0, range=2.0)
```
"""
function MaternModel(discretization::F; smoothness::S, alg = CHOLMODFactorization(), constraint = nothing, observation_points = nothing) where {F <: FEMDiscretization, S <: Integer}
    smoothness >= 0 || throw(ArgumentError("Smoothness must be non-negative, got smoothness=$smoothness"))
    n = ndofs(discretization)
    processed_constraint = _process_constraint(constraint, n)
    # Assemble the κ-independent FEM matrices once; reused on every precision_matrix call.
    C, G = assemble_matern_C_G(discretization)
    # Precompute the κ-invariant structural precision pattern (#183); every
    # precision_matrix call is padded to it.
    D = ndim(discretization)
    α_val = Integer(smoothness_to_ν(smoothness, D) + D // 2)
    Q_pattern = _matern_structural_pattern(
        G, discretization.constraint_handler, discretization.constraint_noise, α_val
    )
    fem_matrices = (; C, G, Q_pattern)
    return MaternModel{F, S, typeof(alg), typeof(processed_constraint), typeof(observation_points), typeof(fem_matrices)}(discretization, smoothness, alg, processed_constraint, observation_points, fem_matrices)
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
    return (τ = Real, range = Real)
end

function _validate_matern_parameters(; τ::Real, range::Real)
    τ > 0 || throw(ArgumentError("Precision parameter τ must be positive, got τ=$τ"))
    range > 0 || throw(ArgumentError("Range parameter must be positive, got range=$range"))
    return nothing
end

function precision_matrix(model::MaternModel{F, S}; τ::Real, range::Real, kwargs...) where {F, S}
    _validate_matern_parameters(; τ = τ, range = range)
    D = ndim(model.discretization)
    ν = smoothness_to_ν(model.smoothness, D)
    κ = range_to_κ(range, ν)
    (; C, G, Q_pattern) = model.fem_matrices
    Q_unscaled = matern_precision_only(model.discretization, model.smoothness, κ, C, G)
    # Scatter τ·Q into the fixed structural pattern instead of `τ * Q_unscaled`:
    # sparse scalar `*` drops fill entries that cancel to exact 0.0 at the
    # current range but not at others, which made the stored pattern
    # range-dependent and broke fixed-pattern workspaces (#183).
    return Symmetric(_pad_scaled_to_pattern(parent(Q_unscaled), τ, Q_pattern))
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

"""
    Matern(discretization::FEMDiscretization; smoothness = 1, alg = CHOLMODFactorization(), constraint = nothing)

Construct a `Matern` formula functor from a pre-built `FEMDiscretization`.

Only available when the `GaussianMarkovRandomFieldsFEM` extension is loaded.
"""
function GaussianMarkovRandomFields.Matern(
        discretization::FEMDiscretization;
        smoothness::Integer = 1,
        alg = CHOLMODFactorization(),
        constraint = nothing,
    )
    smoothness >= 0 || throw(ArgumentError("Smoothness must be non-negative, got smoothness=$smoothness"))
    return GaussianMarkovRandomFields.Matern{
        typeof(discretization), typeof(smoothness), typeof(alg), typeof(constraint),
    }(discretization, smoothness, 1, alg, constraint)
end
