x_left, x_right = -1.0, 1.0
Nₓ = 201
t_start, t_stop = 0.0, 1.0
Nₜ = 101
ts = range(t_start, t_stop, length = Nₜ)
f_initial = x -> exp(-(x + 0.6)^2 / 0.2^2)
xs_initial = range(x_left, x_right, length = Nₓ ÷ 2)
ys_initial = f_initial.(xs_initial)
noise_precision_initial = 0.1^(-2)

x_later = -0.25
y_later = 0.55
noise_precision_later = 0.01^(-2)

xs_all = [xs_initial; x_later]
ys_all = [ys_initial; y_later]
N_obs_all = length(ys_all)

using GMRFs
using Ferrite

grid = generate_grid(Line, (Nₓ - 1,), Tensors.Vec(x_left), Tensors.Vec(x_right))
interpolation = Lagrange{RefLine,1}()
quadrature_rule = QuadratureRule{RefLine}(2)
disc = FEMDiscretization(grid, interpolation, quadrature_rule)

spde_space = MaternSPDE{1}(range = 0.2, smoothness = 1, σ² = 0.3)
spde_time = MaternSPDE{1}(range = 0.5, smoothness = 1)

x_space = discretize(spde_space, disc)
Q_s = precision_map(x_space)

grid_time = generate_grid(Line, (Nₜ - 1,), Tensors.Vec(t_start), Tensors.Vec(t_stop))
disc_time = FEMDiscretization(grid_time, interpolation, quadrature_rule)
x_time = discretize(spde_time, disc_time)
Q_t = precision_map(x_time)

x_st_kron = kronecker_product_spatiotemporal_model(Q_t, Q_s, disc)

A_initial = evaluation_matrix(disc, [Tensors.Vec(x) for x in xs_initial])
t_initial_idx = 1 # Observe at first time point
A_initial = spatial_to_spatiotemporal(A_initial, t_initial_idx, Nₜ)
A_later = evaluation_matrix(disc, [Tensors.Vec(x_later)])
t_later_idx = 2 * Nₜ ÷ 3
A_later = spatial_to_spatiotemporal(A_later, t_later_idx, Nₜ)

A_all = [A_initial; A_later]

using LinearAlgebra, SparseArrays
Q_noise = sparse(I, N_obs_all, N_obs_all) * noise_precision_initial
Q_noise[end, end] = noise_precision_later

x_st_kron_posterior = condition_on_observations(x_st_kron, A_all, Q_noise, ys_all)

using CairoMakie
CairoMakie.activate!()
plot(x_st_kron_posterior, t_initial_idx)

plot(x_st_kron_posterior, Nₜ ÷ 3)

plot(x_st_kron_posterior, 2 * Nₜ ÷ 3)

plot(x_st_kron_posterior, Nₜ)

adv_diff_spde = AdvectionDiffusionSPDE{1}(
    γ = [-0.6],
    H = 0.1 * sparse(I, (1, 1)),
    τ = 0.1,
    α = 2 // 1,
    spatial_spde = spde_space,
    initial_spde = spde_space,
)

x_adv_diff = discretize(adv_diff_spde, disc, ts)

x_adv_diff_posterior = condition_on_observations(x_adv_diff, A_all, Q_noise, ys_all)

plot(x_adv_diff_posterior, t_initial_idx)

plot(x_adv_diff_posterior, Nₜ ÷ 3)

plot(x_adv_diff_posterior, 2 * Nₜ ÷ 3)

plot(x_adv_diff_posterior, Nₜ)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
