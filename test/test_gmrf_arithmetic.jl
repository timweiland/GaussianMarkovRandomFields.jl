using GMRFs, Ferrite, Random, LinearAlgebra, SparseArrays

@testset "GMRF arithmetic" begin
    spde = MaternSPDE{2}(κ=1.0, ν=1)
    grid = generate_grid(Triangle, (20, 20))
    ip = Lagrange{RefTriangle,1}()
    qr = QuadratureRule{RefTriangle}(2)
    disc = FEMDiscretization(grid, ip, qr)

    x = discretize(spde, disc)

    rng = MersenneTwister(6598120)
    @testset "Mean addition" begin
        v = randn(rng, length(x.mean))
        x1 = x + v
        x2 = v + x
        x3 = x - v
        @test x1.mean ≈ x.mean + v
        @test x2.mean ≈ x.mean + v
        @test x3.mean ≈ x.mean - v
        @test x1.precision === x.precision
        @test x2.precision === x.precision
        @test x3.precision === x.precision
    end

    @testset "Joint GMRF" begin
        A = sprand(rng, 5, length(x.mean), 0.3)
        Q_ϵ = sparse(1e8 * I, 5, 5)
        b = randn(rng, 5)
        x1 = x
        x2 = joint_gmrf(x1, A, Q_ϵ, b)
        @test x2.mean ≈ [x1.mean; A * x1.mean + b]
        @test sparse(x2.precision) ≈ [
            sparse(x1.precision)+A'*Q_ϵ*A -A'*Q_ϵ
            -Q_ϵ*A Q_ϵ
        ]
    end

    @testset "GMRF posterior" begin
        A = sprand(rng, 5, length(x.mean), 0.3)
        Q_ϵ = sparse(1e12 * I, 5, 5)
        y = randn(rng, 5)
        b = randn(rng, 5)
        x1 = x
        x2 = condition_on_observations(x1, A, Q_ϵ, y, b)
        @test A * mean(x2) ≈ y - b
        @test mean(var(x2)) < mean(var(x1))
    end
end
