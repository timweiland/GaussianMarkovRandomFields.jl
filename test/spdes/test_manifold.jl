using GaussianMarkovRandomFields
using Ferrite, FerriteGmsh, Gmsh
using LinearAlgebra
using SparseArrays
using Statistics
using Random

# Surface mesh of the unit sphere: 2D triangles embedded in 3D.
function _sphere_test_grid(mesh_size)
    Gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 2)
    gmsh.model.add("sphere_test")
    gmsh.model.occ.addSphere(0.0, 0.0, 0.0, 1.0)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.renumberNodes()
    gmsh.model.mesh.renumberElements()
    nodes = FerriteGmsh.tonodes()
    elements, _ = FerriteGmsh.toelements(2)
    Gmsh.finalize()
    return Ferrite.Grid(elements, nodes)
end

# Exact covariance of the Matérn SPDE solution on the unit sphere between
# points at angle θ = acos(costheta), via the spherical-harmonics series.
function _sphere_matern_cov(costheta, κ, α; lmax = 2000)
    Plm1, Pl = one(costheta), costheta
    s = (1 / (4π)) * (κ^2)^(-α)
    s += (3 / (4π)) * (κ^2 + 2)^(-α) * costheta
    for l in 2:lmax
        Plp = ((2l - 1) * costheta * Pl - (l - 1) * Plm1) / l
        Plm1, Pl = Pl, Plp
        s += ((2l + 1) / (4π)) * (κ^2 + l * (l + 1))^(-α) * Pl
    end
    return s
end

@testset "Matérn fields on manifolds (sphere)" begin
    grid = _sphere_test_grid(0.25)
    ip = Lagrange{RefTriangle, 1}()
    qr = QuadratureRule{RefTriangle}(2)
    disc = FEMDiscretization(grid, ip, qr)
    N_nodes = Ferrite.getnnodes(grid)
    S = node_selection_matrix(disc, 1:N_nodes)

    @testset "Dimensions" begin
        @test ndim(disc) == 3
        @test intrinsic_dim(disc) == 2
    end

    @testset "MaternModel matches the exact sphere covariance" begin
        range_param = 0.5
        model = MaternModel(disc; smoothness = 0)  # ν = 1, α = 2 on a 2-manifold
        gmrf = model(τ = 1.0, range = range_param)

        # The marginal variance is approximately the target σ² = 1
        v = var(gmrf)
        @test 0.75 < sum(v) / length(v) < 1.3

        # One column of the covariance matrix vs. the analytic series
        ν = 1
        κ = sqrt(8ν) / range_param
        σ²_natural = 1 / (4π * κ^2)  # Euclidean Matérn variance for ν = 1, d = 2
        Q = precision_matrix(model; τ = 1.0, range = range_param)
        ref_dof = findfirst(!iszero, Vector(S[1, :]))
        e = zeros(size(Q, 1))
        e[ref_dof] = 1.0
        cov_column = cholesky(sparse(Q)) \ e
        x_ref = grid.nodes[1].x
        cov_exact = [
            _sphere_matern_cov(
                    clamp(x_ref ⋅ n.x / (norm(x_ref) * norm(n.x)), -1, 1), κ, 2
                ) / σ²_natural
                for n in grid.nodes
        ]
        rel_err = norm(S * cov_column - cov_exact) / norm(cov_exact)
        @test rel_err < 0.15
    end

    @testset "discretize(MaternSPDE{2}) agrees with MaternModel" begin
        spde = MaternSPDE{2}(range = 0.5, smoothness = 0)
        x = discretize(spde, disc)
        Q_spde = to_matrix(precision_map(x))
        model = MaternModel(disc; smoothness = 0)
        Q_model = precision_matrix(model; τ = 1.0, range = 0.5)
        @test norm(sparse(Q_spde) - sparse(Q_model)) / norm(sparse(Q_model)) < 1.0e-10
    end

    @testset "SPDE dimension must match the intrinsic dimension" begin
        @test_throws ArgumentError discretize(MaternSPDE{3}(range = 0.5, smoothness = 0), disc)
        @test_throws ArgumentError discretize(
            AdvectionDiffusionSPDE{3}(γ = [0.0, 0.0, 0.0]), disc, range(0.0, 1.0, length = 3)
        )
    end

    @testset "Anisotropic diffusion requires embedding coordinates" begin
        spde_bad = MaternSPDE{2}(
            range = 0.5, smoothness = 0, diffusion_factor = [1.0 0.0; 0.0 2.0]
        )
        @test_throws ArgumentError discretize(spde_bad, disc)
        # ... but a uniform scaling is expanded automatically
        spde_ok = MaternSPDE{2}(
            range = 0.5, smoothness = 0, diffusion_factor = 0.5 * Matrix(I, 2, 2)
        )
        x = discretize(spde_ok, disc)
        @test size(to_matrix(precision_map(x))) == (ndofs(disc), ndofs(disc))
    end

    @testset "Point evaluation projects onto the surface mesh" begin
        rng = MersenneTwister(42)
        pts = [normalize(randn(rng, 3)) for _ in 1:20]
        A = evaluation_matrix(disc, [Ferrite.Vec(p...) for p in pts])
        @test size(A) == (20, ndofs(disc))
        # Points on the true sphere are slightly off the faceted surface, yet
        # each row is a valid convex combination of P1 weights.
        @test all(isapprox.(sum(A, dims = 2), 1.0, atol = 1.0e-8))
        @test all(A .>= 0)

        # The matrix-input convenience method agrees
        A_mat = evaluation_matrix(disc, permutedims(hcat(pts...)))
        @test A_mat ≈ A

        # Evaluation at mesh nodes selects exactly the nodal DOFs
        node_pts = [Ferrite.Vec(grid.nodes[i].x...) for i in 1:5]
        @test evaluation_matrix(disc, node_pts) ≈ S[1:5, :]

        # Points far away from the surface are rejected
        @test_throws ArgumentError evaluation_matrix(disc, [Ferrite.Vec(0.0, 0.0, 0.0)])
        @test_throws ArgumentError evaluation_matrix(disc, [Ferrite.Vec(0.0, 0.0, 2.0)])
    end

    @testset "Advection-diffusion with a tangential velocity field" begin
        # Solid-body rotation about the z-axis: γ(x) = ω × x
        Ω = π
        wind(x) = Ferrite.Vec(-Ω * x[2], Ω * x[1], 0.0)
        ts = range(0.0, 0.5, length = 17)
        spde = AdvectionDiffusionSPDE{2}(
            κ = 0.1, α = 1 // 1, H = sparse(0.01 * I, 2, 2), γ = wind, τ = 0.1
        )
        X = discretize(spde, disc, ts)
        @test length(mean(X)) == ndofs(disc) * length(ts)

        # Observe a bump on the equator at the initial time; the posterior
        # mean's peak should be transported by ≈ Ω * T radians of azimuth.
        p0 = [1.0, 0.0, 0.0]
        y = [exp(-acos(clamp(n.x ⋅ p0, -1, 1))^2 / (2 * 0.4^2)) for n in grid.nodes]
        A_bump = spatial_to_spatiotemporal(S, 1, length(ts))
        X_post = linear_condition(X; A = A_bump, Q_ϵ = 1.0e6, y = y)
        means = time_means(X_post)

        azimuth(node_idx) = atan(grid.nodes[node_idx].x[2], grid.nodes[node_idx].x[1])
        peak_start = argmax(S * means[1])
        peak_end = argmax(S * means[end])
        @test abs(azimuth(peak_start)) < 0.3
        drift = rem(azimuth(peak_end) - azimuth(peak_start) - Ω * 0.5, 2π, RoundNearest)
        @test abs(drift) < 0.5

        # Constant advection vectors must be given in embedding coordinates
        X_const = discretize(
            AdvectionDiffusionSPDE{2}(
                κ = 0.1, α = 1 // 1, H = sparse(0.01 * I, 2, 2),
                γ = [0.5, 0.0, 0.0], τ = 0.1,
            ),
            disc, ts,
        )
        @test length(mean(X_const)) == ndofs(disc) * length(ts)

        # Streamline diffusion is not available for velocity *fields*
        @test_throws ArgumentError discretize(spde, disc, ts; streamline_diffusion = true)
    end
end
