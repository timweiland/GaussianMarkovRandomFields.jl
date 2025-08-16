import Random: MersenneTwister

using Ferrite
using GaussianMarkovRandomFields
using LinearAlgebra
using SparseArrays

@testset "Advection-Diffusion Prior (streamline diffusion: $sd)" for sd ∈ [false, true]
    rng = MersenneTwister(364802394)
    grid = generate_grid(Line, (50,))
    ip = Lagrange{RefLine,1}()
    qr = QuadratureRule{RefLine}(2)
    disc = FEMDiscretization(grid, ip, qr)
    disc_periodic = FEMDiscretization(
        grid,
        ip,
        qr,
        [(:u, nothing)],
        [(_get_periodic_constraint(grid), 1e-4)],
    )
    disc_dirichlet = FEMDiscretization(
        grid,
        ip,
        qr,
        [(:u, nothing)],
        [(_get_dirichlet_constraint(grid, 0.0, 0.0), 1e-4)],
    )


    ts = 0:0.05:1

    κₛ = range_to_κ(0.25, 1 // 2)

    κ_slow = 0.05
    H_slow = 0.05 * sparse(I, (1, 1))
    κ_fast = 10 * κ_slow
    H_fast = 10 * H_slow
    τ = 0.1
    γ_left = [0.25]
    γ_right = [-0.25]
    γ_static = [0.0]



    spde_fast_right = AdvectionDiffusionSPDE{1}(
        κ = κ_fast,
        H = H_fast,
        γ = γ_right,
        τ = τ,
        initial_spde = MaternSPDE{1}(κ = κₛ, smoothness = 1, diffusion_factor = H_fast),
        spatial_spde = MaternSPDE{1}(κ = κₛ, smoothness = 1, diffusion_factor = H_fast),
    )
    spde_fast_left = AdvectionDiffusionSPDE{1}(
        κ = κ_fast,
        H = H_fast,
        γ = γ_left,
        τ = τ,
        initial_spde = MaternSPDE{1}(κ = κₛ, smoothness = 1, diffusion_factor = H_fast),
        spatial_spde = MaternSPDE{1}(κ = κₛ, smoothness = 1, diffusion_factor = H_fast),
    )
    spde_fast_static = AdvectionDiffusionSPDE{1}(
        κ = κ_fast,
        H = H_fast,
        γ = γ_static,
        τ = τ,
        initial_spde = MaternSPDE{1}(κ = κₛ, smoothness = 2, diffusion_factor = H_fast),
        spatial_spde = MaternSPDE{1}(κ = κₛ, smoothness = 1, diffusion_factor = H_fast),
    )
    spde_slow_static = AdvectionDiffusionSPDE{1}(
        κ = κ_slow,
        H = H_slow,
        γ = γ_static,
        τ = τ,
        initial_spde = MaternSPDE{1}(κ = κₛ, smoothness = 2, diffusion_factor = H_slow),
        spatial_spde = MaternSPDE{1}(κ = κₛ, smoothness = 1, diffusion_factor = H_slow),
    )
    x_prior_fast_right = GaussianMarkovRandomFields.discretize(spde_fast_right, disc, ts; streamline_diffusion = sd)
    x_prior_fast_left = GaussianMarkovRandomFields.discretize(spde_fast_left, disc, ts; streamline_diffusion = sd)
    x_prior_fast_static = GaussianMarkovRandomFields.discretize(spde_fast_static, disc, ts; streamline_diffusion = sd)
    x_prior_slow_static = GaussianMarkovRandomFields.discretize(spde_slow_static, disc, ts; streamline_diffusion = sd)

    xs = range(-1.0, 1.0, length = 100)
    spread = 0.2
    ic_fn = x -> exp(-x^2 / (2 * spread^2))
    ys = ic_fn.(xs)

    A_ic = evaluation_matrix(disc, [Tensors.Vec(x) for x in xs])
    A_last = spatial_to_spatiotemporal(A_ic, length(ts), length(ts))
    A_ic = spatial_to_spatiotemporal(A_ic, 1, length(ts))

    get_peak_initial = x_posterior -> xs[argmax(A_ic * mean(x_posterior))]
    get_peak_final = x_posterior -> xs[argmax(A_last * mean(x_posterior))]

    @testset "Advection" begin
        x_cond_fast_right = condition_on_observations(x_prior_fast_right, A_ic, 1e8, ys)
        x_cond_fast_left = condition_on_observations(x_prior_fast_left, A_ic, 1e8, ys)
        x_cond_fast_static = condition_on_observations(x_prior_fast_static, A_ic, 1e8, ys)

        peak_right_initial = get_peak_initial(x_cond_fast_right)
        peak_right_final = get_peak_final(x_cond_fast_right)
        peak_left_initial = get_peak_initial(x_cond_fast_left)
        peak_left_final = get_peak_final(x_cond_fast_left)
        peak_static_initial = get_peak_initial(x_cond_fast_static)
        peak_static_final = get_peak_final(x_cond_fast_static)
        @test peak_right_initial ≈ 0.0 atol = 0.1
        @test peak_right_final > peak_right_initial + 0.1
        @test peak_left_initial ≈ 0.0 atol = 0.1
        @test peak_left_final < peak_left_initial - 0.1
        @test peak_static_initial ≈ 0.0 atol = 0.1
        @test peak_static_final ≈ 0.0 atol = 0.1
    end

    @testset "Diffusion" begin
        x_cond_slow_static = condition_on_observations(x_prior_slow_static, A_ic, 1e8, ys)
        x_cond_fast_static = condition_on_observations(x_prior_fast_static, A_ic, 1e8, ys)

        final_vals_slow = A_last * mean(x_cond_slow_static)
        final_vals_fast = A_last * mean(x_cond_fast_static)

        @test maximum(final_vals_slow) > maximum(final_vals_fast)
    end

    @testset "Boundary conditions" begin
        x_prior_fast_right_periodic =
            GaussianMarkovRandomFields.discretize(spde_fast_right, disc_periodic, ts; streamline_diffusion = sd)
        x_cond_fast_right_periodic =
            condition_on_observations(x_prior_fast_right_periodic, A_ic, 1e8, ys)

        x_prior_fast_right_dirichlet =
            GaussianMarkovRandomFields.discretize(spde_fast_right, disc_dirichlet, ts; streamline_diffusion = sd)
        x_cond_fast_right_dirichlet =
            condition_on_observations(x_prior_fast_right_dirichlet, A_ic, 1e8, ys)

        N_samples = 3
        samples_periodic = [time_rands(x_cond_fast_right_periodic, rng) for _ = 1:N_samples]
        samples_dirichlet =
            [time_rands(x_cond_fast_right_dirichlet, rng) for _ = 1:N_samples]
        for samp_idx = 1:N_samples
            for t in eachindex(ts)
                @test samples_periodic[samp_idx][t][1] ≈ samples_periodic[samp_idx][t][end] atol =
                    1e-1
                @test samples_dirichlet[samp_idx][t][1] ≈ 0.0 atol = 1e-1
                @test samples_dirichlet[samp_idx][t][end] ≈ 0.0 atol = 1e-1
            end
        end
    end
end
