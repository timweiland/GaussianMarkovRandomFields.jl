using LinearAlgebra, Ferrite

export kronecker_product_spatiotemporal_model

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
