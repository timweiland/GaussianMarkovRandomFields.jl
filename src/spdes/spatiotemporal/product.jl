using LinearAlgebra, Ferrite

export kronecker_product_spatiotemporal_model

"""
    kronecker_product_spatiotemporal_model(
        Q_t::AbstractMatrix,
        Q_s::AbstractMatrix,
        spatial_disc::FEMDiscretization;
        algorithm = nothing,
    )

Create a spatiotemporal GMRF through a Kronecker product of the temporal and
spatial precision matrices.

# Arguments
- `Q_t::AbstractMatrix`: The temporal precision matrix.
- `Q_s::AbstractMatrix`: The spatial precision matrix.
- `spatial_disc::FEMDiscretization`: The spatial discretization.

# Keyword arguments
- `algorithm`: The LinearSolve algorithm to use.
"""
function kronecker_product_spatiotemporal_model(
        Q_t::Union{LinearMap, AbstractMatrix},
        Q_s::Union{LinearMap, AbstractMatrix},
        spatial_disc::FEMDiscretization;
        algorithm = nothing,
    )
    Q_st = kron(Q_t, Q_s)
    return ConcreteConstantMeshSTGMRF(
        GMRF(
            zeros(Base.size(Q_st, 1)),
            Q_st,
            algorithm,
        ),
        spatial_disc,
    )
end
