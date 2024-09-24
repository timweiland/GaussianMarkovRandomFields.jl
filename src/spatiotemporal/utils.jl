export spatial_to_spatiotemporal

@views function make_chunks(X::AbstractVector, n::Integer)
    c = length(X) ÷ n
    return [X[1+c*k:(k == n - 1 ? end : c * k + c)] for k = 0:n-1]
end

"""
    spatial_to_spatiotemporal(spatial_matrix, t_idx, N_t)

Make a spatial matrix applicable to a spatiotemporal system at time index `t_idx`.
Results in a matrix that selects the spatial information exactly at time `t_idx`.
"""
function spatial_to_spatiotemporal(spatial_matrix::AbstractMatrix, t_idx, N_t)
    E_t = spzeros(1, N_t)
    E_t[t_idx] = 1
    return kron(E_t, spatial_matrix)
end
