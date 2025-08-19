# Adding a deterministic vector to a GMRF
Base.:+(d::GMRF, b::AbstractVector) = GMRF(d.mean + b, d.precision)
Base.:+(b::AbstractVector, d::GMRF) = d + b
Base.:-(d::GMRF, b::AbstractVector) = GMRF(d.mean - b, d.precision)

# Adding a deterministic vector to a MetaGMRF
Base.:+(d::MetaGMRF, b::AbstractVector) = MetaGMRF(d.gmrf + b, d.metadata)
Base.:+(b::AbstractVector, d::MetaGMRF) = d + b
Base.:-(d::MetaGMRF, b::AbstractVector) = MetaGMRF(d.gmrf - b, d.metadata)
