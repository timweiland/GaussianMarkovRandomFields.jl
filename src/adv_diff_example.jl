using GaussianMarkovRandomFields, Ferrite, LinearAlgebra, SparseArrays

grid = generate_grid(Triangle, (30, 30))
ip = Lagrange{2,RefTetrahedron,1}()
qr = QuadratureRule{2,RefTetrahedron}(2)
disc = FEMDiscretization(grid, ip, qr)

spde = AdvectionDiffusionSPDE{2}(1.0, 1 // 1, [1.0 0.0; 0.0 1.0], [0.0; 0.0], 1.0, 50.0)
ts = 0:0.005:1

X = discretize(spde, disc, ts; streamline_diffusion = false, κ_matern = 5.0)

xs = -0.95:0.2:0.95
ys = -0.95:0.2:0.95
D = reshape([Tensors.Vec(x, y) for x ∈ xs, y ∈ ys], length(xs) * length(ys))

E_spatial = evaluation_matrix(disc, D)
E = spatial_to_spatiotemporal(E_spatial, 1, length(ts))
ic_vals = [10 * exp(-(x[1]^2 + x[2]^2) / 0.1) for x ∈ D]
N_ic = length(ic_vals)
Q_ϵ = sparse(1e10 * I, (N_ic, N_ic))

E₂ = spatial_to_spatiotemporal(E_spatial, 100, length(ts))
ic_vals₂ = [10 * exp(-((x[1] - 0.3)^2 + (x[2] - 0.3)^2) / 0.1) for x ∈ D]

ic_vals_total = [ic_vals; ic_vals₂]
E_total = [E; E₂]


N_ic = length(ic_vals_total)
Q_ϵ = sparse(1e10 * I, (N_ic, N_ic))
X_cond = condition_on_observations(X, E_total, Q_ϵ, ic_vals_total)

# X₀ = X.ssm.x₀
# X₀_cond = condition_on_observations(X₀, E_spatial, Q_ϵ, ic_vals)
# pred₀ = Array(mean(X₀_cond))
# G = G = sparse(X.ssm.G(0.005))
# M = sparse(X.ssm.M(0.005))

# preds = [pred₀]
# for t in ts[2:end]
#     pred = G \ preds[end]
#     pred = M * pred
#     push!(preds, pred)
# end
# full_pred = vcat(preds...)
