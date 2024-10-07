using LinearAlgebra, Ferrite

export kronecker_product_spatiotemporal_model, product_matern

function kronecker_product_spatiotemporal_model(
    Q_t::AbstractMatrix,
    Q_s::AbstractMatrix,
    spatial_disc::FEMDiscretization;
    solver_blueprint=DefaultSolverBlueprint(),
)
    Q = kron(Q_t, Q_s)
    return ConcreteConstantMeshSTGMRF(
        zeros(size(Q, 1)),
        LinearMap(Q),
        spatial_disc,
        solver_blueprint
    )
end

function product_matern(
    matern_temporal::MaternSPDE,
    N_t::Int,
    matern_spatial::MaternSPDE,
    spatial_disc::FEMDiscretization;
    solver_blueprint=DefaultSolverBlueprint(),
)
    offset = N_t ÷ 10
    temporal_grid = generate_grid(Line, (N_t + 2*offset - 1,))
    temporal_ip = Lagrange{1, RefCube, 1}()
    temporal_qr = QuadratureRule{1, RefCube}(2)
    temporal_disc = FEMDiscretization(temporal_grid, temporal_ip, temporal_qr)
    x_t = discretize(matern_temporal, temporal_disc)

    Q_t = to_matrix(precision_map(x_t))[offset+1:end-offset, offset+1:end-offset]
    x_s = discretize(matern_spatial, spatial_disc)
    Q_s = to_matrix(precision_map(x_s))

    x_spatiotemporal = kronecker_product_spatiotemporal_model(Q_t, Q_s, spatial_disc; solver_blueprint=solver_blueprint)
    if length(disc.constraint_handler.prescribed_dofs) > 0
        return ConstrainedGMRF(x_spatiotemporal, disc.constraint_handler)
    end
end