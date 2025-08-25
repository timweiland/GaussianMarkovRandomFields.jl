using Test
using GaussianMarkovRandomFields
using Ferrite
using LinearAlgebra
using SparseArrays

@testset "Constant Mesh Spatiotemporal GMRF" begin

    @testset "ConcreteConstantMeshSTGMRF Construction" begin
        # Create a simple spatial discretization
        grid = generate_grid(Line, (4,))
        ip = Lagrange{RefLine, 1}()
        qr = QuadratureRule{RefLine}(2)
        discretization = FEMDiscretization(grid, ip, qr)

        n_spatial = ndofs(discretization)
        n_t = 3

        # Create GMRF for spatiotemporal system
        total_size = n_spatial * n_t
        mean_vec = zeros(total_size)
        precision_mat = sparse(I, total_size, total_size)
        gmrf = GMRF(mean_vec, precision_mat)

        # Test constructor
        st_gmrf = ConcreteConstantMeshSTGMRF(gmrf, discretization)

        @test st_gmrf isa ConcreteConstantMeshSTGMRF
        @test GaussianMarkovRandomFields.N_spatial(st_gmrf) == n_spatial
        @test GaussianMarkovRandomFields.N_t(st_gmrf) == n_t
        @test discretization_at_time(st_gmrf, 1) === discretization

        # Test constructor with size mismatch
        wrong_gmrf = GMRF(zeros(n_spatial + 1), sparse(I, n_spatial + 1, n_spatial + 1))
        @test_throws ArgumentError ConcreteConstantMeshSTGMRF(wrong_gmrf, discretization)
    end

    @testset "Spatiotemporal utility functions" begin
        # Simple setup
        grid = generate_grid(Triangle, (3, 2))
        ip = Lagrange{RefTriangle, 1}()
        qr = QuadratureRule{RefTriangle}(1)
        discretization = FEMDiscretization(grid, ip, qr)

        n_spatial = ndofs(discretization)
        n_t = 4
        total_size = n_spatial * n_t

        # Create GMRF with ordered values for testing
        mean_vec = collect(1.0:total_size)
        precision_mat = sparse(I, total_size, total_size)
        gmrf = GMRF(mean_vec, precision_mat)

        st_gmrf = ConcreteConstantMeshSTGMRF(gmrf, discretization)

        # Test time_means chunking
        time_mean_chunks = time_means(st_gmrf)
        @test length(time_mean_chunks) == n_t
        @test all(length.(time_mean_chunks) .== n_spatial)

        # Verify first chunk
        @test time_mean_chunks[1] == collect(1.0:n_spatial)
        # Verify last chunk
        @test time_mean_chunks[end] == collect((total_size - n_spatial + 1):total_size)

        # Test other utility functions return correct structure
        @test length(time_vars(st_gmrf)) == n_t
        @test length(time_stds(st_gmrf)) == n_t

        # Test time_rands
        using Random
        rng = MersenneTwister(123)
        rand_chunks = time_rands(st_gmrf, rng)
        @test length(rand_chunks) == n_t
        @test all(length.(rand_chunks) .== n_spatial)
    end

    @testset "Default preconditioner strategy" begin
        grid = generate_grid(Line, (5,))
        ip = Lagrange{RefLine, 1}()
        qr = QuadratureRule{RefLine}(2)
        discretization = FEMDiscretization(grid, ip, qr)

        n_spatial = ndofs(discretization)
        n_t = 2
        total_size = n_spatial * n_t

        # Create simple block structure
        precision_mat = sparse(I, total_size, total_size)
        gmrf = GMRF(zeros(total_size), precision_mat)
        st_gmrf = ConcreteConstantMeshSTGMRF(gmrf, discretization)

        # Test preconditioner can be created
        preconditioner = GaussianMarkovRandomFields.default_preconditioner_strategy(st_gmrf)
        @test preconditioner isa GaussianMarkovRandomFields.AbstractPreconditioner
    end
end
