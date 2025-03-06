using LinearAlgebra, Ferrite

export kronecker_product_spatiotemporal_model

"""
    kronecker_product_spatiotemporal_model(
        Q_t::AbstractMatrix,
        Q_s::AbstractMatrix,
        spatial_disc::FEMDiscretization;
        solver_blueprint = DefaultSolverBlueprint(),
    )

Create a spatiotemporal GMRF through a Kronecker product of the temporal and
spatial precision matrices.

# Arguments
- `Q_t::AbstractMatrix`: The temporal precision matrix.
- `Q_s::AbstractMatrix`: The spatial precision matrix.
- `spatial_disc::FEMDiscretization`: The spatial discretization.

# Keyword arguments
- `solver_blueprint::AbstractSolverBlueprint=DefaultSolverBlueprint()`:
        The solver blueprint.
"""
function kronecker_product_spatiotemporal_model(
    Q_t::AbstractMatrix,
    Q_s::AbstractMatrix,
    spatial_disc::FEMDiscretization;
    solver_blueprint = DefaultSolverBlueprint(),
)
    Q = kron(Q_t, Q_s)
    return ConcreteConstantMeshSTGMRF(
        zeros(Base.size(Q, 1)),
        LinearMap(Q),
        spatial_disc,
        solver_blueprint,
    )
end
