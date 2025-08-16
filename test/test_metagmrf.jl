using GaussianMarkovRandomFields
using LinearAlgebra, SparseArrays, Random, Distributions
using Test

@testset "MetaGMRF Tests" begin
    # Setup base GMRF for testing
    n = 10
    Q = spdiagm(0 => 2*ones(n), 1 => -ones(n-1), -1 => -ones(n-1))
    μ = zeros(n)
    gmrf = GMRF(μ, Q)
    
    # Test metadata type
    struct TestMetadata <: GMRFMetadata
        name::String
        value::Int
    end
    
    metadata = TestMetadata("test", 42)
    meta_gmrf = MetaGMRF(gmrf, metadata)
    
    @testset "Construction and Type System" begin
        @test isa(meta_gmrf, MetaGMRF{TestMetadata})
        @test isa(meta_gmrf, AbstractGMRF)
        @test meta_gmrf.gmrf === gmrf
        @test meta_gmrf.metadata === metadata
        @test meta_gmrf.metadata.name == "test"
        @test meta_gmrf.metadata.value == 42
    end
    
    @testset "Core GMRF Operations" begin
        # Basic operations
        @test length(meta_gmrf) == length(gmrf)
        @test mean(meta_gmrf) == mean(gmrf)
        @test precision_map(meta_gmrf) === precision_map(gmrf)
        @test precision_matrix(meta_gmrf) == precision_matrix(gmrf)
        @test information_vector(meta_gmrf) == information_vector(gmrf)
        @test var(meta_gmrf) == var(gmrf)
        @test std(meta_gmrf) == std(gmrf)
        
        # Random generation
        Random.seed!(123)
        sample1 = rand(meta_gmrf)
        Random.seed!(123)
        sample2 = rand(gmrf)
        @test sample1 == sample2
        @test length(sample1) == n
        
        Random.seed!(456)
        rng = MersenneTwister(456)
        sample3 = rand(rng, meta_gmrf)
        Random.seed!(456)
        rng2 = MersenneTwister(456)
        sample4 = rand(rng2, gmrf)
        @test sample3 == sample4
    end
    
    @testset "Distributions.jl Methods" begin
        x = randn(n)
        
        # Test forwarding of Distributions methods
        @test invcov(meta_gmrf) == invcov(gmrf)
        @test logdetcov(meta_gmrf) == logdetcov(gmrf)
        @test sqmahal(meta_gmrf, x) == sqmahal(gmrf, x)
        @test gradlogpdf(meta_gmrf, x) == gradlogpdf(gmrf, x)
        
        # Test in-place sqmahal
        r1 = zeros(n)
        r2 = zeros(n)
        sqmahal!(r1, meta_gmrf, x)
        sqmahal!(r2, gmrf, x)
        @test r1 == r2
    end
    
    @testset "Automatic Wrapper Preservation" begin
        # Create observation setup
        E = sparse([1.0 0 0 zeros(1, 7)...])  # Observe first element
        Q_eps = 1e4
        y = [1.0]
        
        # Test linear_condition preserves wrapper type and metadata
        conditioned = linear_condition(meta_gmrf; A=E, Q_ϵ=Q_eps, y=y)
        @test isa(conditioned, MetaGMRF{TestMetadata})
        @test conditioned.metadata === metadata
        @test conditioned.metadata.name == "test"
        @test conditioned.metadata.value == 42
        
        # Test condition_on_observations preserves wrapper type
        conditioned2 = condition_on_observations(meta_gmrf, E, Q_eps, y)
        @test isa(conditioned2, MetaGMRF{TestMetadata})
        @test conditioned2.metadata === metadata
        
        # Test that conditioning actually changed the mean
        @test abs(mean(conditioned)[1] - 1.0) < 0.1
        @test mean(conditioned) != mean(meta_gmrf)
        
        # Test conditioning chain preserves type
        conditioned3 = condition_on_observations(conditioned2, E, Q_eps, [0.5])
        @test isa(conditioned3, MetaGMRF{TestMetadata})
        @test conditioned3.metadata === metadata
    end
    
    @testset "Show Methods" begin
        # Test compact show
        io = IOBuffer()
        show(io, meta_gmrf)
        output = String(take!(io))
        @test contains(output, "MetaGMRF{TestMetadata}")
        
        # Test full show
        io = IOBuffer()
        show(io, MIME("text/plain"), meta_gmrf)
        output = String(take!(io))
        @test contains(output, "MetaGMRF{TestMetadata}")
        @test contains(output, "Inner GMRF:")
        @test contains(output, "Metadata:")
    end
    
    @testset "Multiple Metadata Types" begin
        # Test different metadata type
        struct SpatialMetadata <: GMRFMetadata
            dims::Tuple{Int,Int}
            coords::Matrix{Float64}
        end
        
        spatial_meta = SpatialMetadata((2, 3), randn(2, 3))
        spatial_gmrf = MetaGMRF(gmrf, spatial_meta)
        
        @test isa(spatial_gmrf, MetaGMRF{SpatialMetadata})
        @test spatial_gmrf.metadata.dims == (2, 3)
        @test size(spatial_gmrf.metadata.coords) == (2, 3)
        
        # Test that different metadata types are distinct
        @test typeof(meta_gmrf) != typeof(spatial_gmrf)
    end
end
