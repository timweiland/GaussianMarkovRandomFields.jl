using GaussianMarkovRandomFields, Ferrite
grid = generate_grid(Line, (50,))
interpolation = Lagrange{RefLine, 1}()
quadrature_rule = QuadratureRule{RefLine}(2)

function get_dirichlet_constraint(grid::Ferrite.Grid{1})
    boundary = getfacetset(grid, "left") ∪ getfacetset(grid, "right")

    return Dirichlet(:u, boundary, x -> (x[1] ≈ -1.0) ? 0.0 : (x[1] ≈ 1.0) ? 0.0 : 0.0)
end

dbc = get_dirichlet_constraint(grid)

bcs = [(dbc, 1.0e-4)] # 1e-4 is the noise in terms of the standard deviation

disc = FEMDiscretization(grid, interpolation, quadrature_rule, [(:u, nothing)], bcs)

matern_spde = MaternSPDE{1}(range = 0.5, smoothness = 1, σ² = 0.3)
x = discretize(matern_spde, disc)

using CairoMakie
CairoMakie.activate!()
plot(x, disc)

function get_periodic_constraint(grid::Ferrite.Grid{1})
    cellidx_left, dofidx_left = collect(grid.facetsets["left"])[1]
    cellidx_right, dofidx_right = collect(grid.facetsets["right"])[1]

    temp_dh = DofHandler(grid)
    add!(temp_dh, :u, Lagrange{RefLine, 1}())
    close!(temp_dh)
    cc = CellCache(temp_dh)
    get_dof(cell_idx, dof_idx) = (reinit!(cc, cell_idx); celldofs(cc)[dof_idx])
    dof_left = get_dof(cellidx_left, dofidx_left)
    dof_right = get_dof(cellidx_right, dofidx_right)

    return AffineConstraint(dof_left, [dof_right => 1.0], 0.0)
end

pbc = get_periodic_constraint(grid)

bcs = [(pbc, 1.0e-4)]
disc_periodic =
    FEMDiscretization(grid, interpolation, quadrature_rule, [(:u, nothing)], bcs)
x_periodic = discretize(matern_spde, disc_periodic)

plot(x_periodic, disc)

using LinearAlgebra, SparseArrays
spde = AdvectionDiffusionSPDE{1}(
    γ = [-0.6],
    H = 0.1 * sparse(I, (1, 1)),
    τ = 0.1,
    α = 2 // 1,
    spatial_spde = matern_spde,
    initial_spde = matern_spde,
)
ts = 0:0.05:1
N_t = length(ts)
x_adv_diff_dirichlet = discretize(spde, disc, ts)
x_adv_diff_periodic = discretize(spde, disc_periodic, ts)

xs_ic = -0.99:0.01:0.99
ys_ic = exp.(-xs_ic .^ 2 / 0.2^2)
A_ic = evaluation_matrix(disc, [Tensors.Vec(x) for x in xs_ic])
A_ic = spatial_to_spatiotemporal(A_ic, 1, N_t)

x_adv_diff_dirichlet = condition_on_observations(x_adv_diff_dirichlet, A_ic, 1.0e8, ys_ic)
x_adv_diff_periodic = condition_on_observations(x_adv_diff_periodic, A_ic, 1.0e8, ys_ic)

plot(x_adv_diff_dirichlet, 1)

plot(x_adv_diff_dirichlet, N_t ÷ 2)

plot(x_adv_diff_dirichlet, N_t)

plot(x_adv_diff_periodic, N_t ÷ 2)

plot(x_adv_diff_periodic, N_t)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
