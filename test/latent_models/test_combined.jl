using SparseArrays
using LinearAlgebra
using GaussianMarkovRandomFields

@testset "CombinedModel" begin

    @testset "Constructor and length" begin
        besag = BesagModel(sparse([0 1; 1 0]))
        iid = IIDModel(3)

        # Vector constructor
        combined1 = CombinedModel([besag, iid])
        @test length(combined1) == 5
        @test combined1.component_sizes == [2, 3]

        # Variadic constructor (syntactic sugar)
        combined2 = CombinedModel(besag, iid)
        @test length(combined2) == 5
        @test combined2.component_sizes == [2, 3]

        @test_throws ArgumentError CombinedModel(LatentModel[])
    end

    @testset "Parameter naming" begin
        # BYM model (Besag + IID)
        W = sparse([0 1 0; 1 0 1; 0 1 0])
        bym = CombinedModel([BesagModel(W), IIDModel(3)])
        params = hyperparameters(bym)
        @test haskey(params, :τ_besag) && haskey(params, :τ_iid)

        # Duplicate models get suffixes
        triple_iid = CombinedModel([IIDModel(2), IIDModel(3), IIDModel(4)])
        params = hyperparameters(triple_iid)
        @test haskey(params, :τ_iid) && haskey(params, :τ_iid_2) && haskey(params, :τ_iid_3)

        # Mixed models
        mixed = CombinedModel([AR1Model(2), RW1Model(3)])
        params = hyperparameters(mixed)
        @test haskey(params, :τ_ar1) && haskey(params, :ρ_ar1) && haskey(params, :τ_rw1)
    end

    @testset "Block diagonal precision matrix" begin
        ar1 = AR1Model(2)
        iid = IIDModel(3)
        combined = CombinedModel([ar1, iid])

        Q = precision_matrix(combined; τ_ar1 = 1.0, ρ_ar1 = 0.5, τ_iid = 2.0)
        @test size(Q) == (5, 5)

        # Check block structure
        Q_ar1 = precision_matrix(ar1; τ = 1.0, ρ = 0.5)
        Q_iid = precision_matrix(iid; τ = 2.0)
        @test Q[1:2, 1:2] ≈ Q_ar1
        @test Q[3:5, 3:5] ≈ Q_iid
        @test all(Q[1:2, 3:5] .== 0)  # Off-diagonal blocks zero
    end

    @testset "Mean and constraints" begin
        W = sparse([0 1; 1 0])
        bym = CombinedModel([BesagModel(W), IIDModel(2)])

        # Mean is always zero for our models
        μ = mean(bym; τ_besag = 1.0, τ_iid = 2.0)
        @test length(μ) == 4 && all(μ .== 0)

        # Mixed constraints: Besag has sum-to-zero, IID has none
        A, e = constraints(bym; τ_besag = 1.0, τ_iid = 2.0)
        @test size(A) == (1, 4)
        @test A[1, 1:2] ≈ [1.0, 1.0] && all(A[1, 3:4] .== 0)

        # No constraints when all components are unconstrained
        unconstrained = CombinedModel([AR1Model(2), IIDModel(2)])
        @test constraints(unconstrained; τ_ar1 = 1.0, ρ_ar1 = 0.5, τ_iid = 1.0) === nothing
    end

    @testset "GMRF construction" begin
        W = sparse([0 1 0; 1 0 1; 0 1 0])
        bym = CombinedModel([BesagModel(W), IIDModel(3)])

        # Constrained due to Besag
        gmrf = bym(τ_besag = 1.0, τ_iid = 2.0)
        @test gmrf isa ConstrainedGMRF && length(gmrf) == 6

        # Unconstrained
        unconstrained = CombinedModel([AR1Model(2), IIDModel(2)])
        gmrf2 = unconstrained(τ_ar1 = 1.0, ρ_ar1 = 0.5, τ_iid = 1.0)
        @test gmrf2 isa GMRF && length(gmrf2) == 4
    end

    @testset "Model name" begin
        @test model_name(CombinedModel([IIDModel(2)])) == :combined
    end
end
