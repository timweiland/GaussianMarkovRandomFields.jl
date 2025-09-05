using ReTest
using StatsModels
using Distributions
using GaussianMarkovRandomFields
using SparseArrays

@testset "Formula Interface (MVP)" begin
    # Common small dataset
    n = 10
    data = (
        y = randn(n),
        group = repeat(1:2, inner = div(n, 2)),
        time = collect(1:n),
        x = randn(n),
    )

    @testset "IID + RW1 (no intercept)" begin
        comp = build_formula_components(@formula(y ~ 0 + IID(group) + RandomWalk(1, time)), data; family = Normal)
        @test size(comp.A) == (n, 2 + n)
        @test nnz(comp.A) == n + n  # one indicator per block per row
        @test comp.meta.n_random == 2
        @test comp.meta.n_fixed == 0
        @test length(comp.combined_model) == 2 + n
        ks = Set(keys(comp.hyperparams))
        @test (:τ_iid in ks) && (:τ_rw1 in ks)
    end

    @testset "AR1 (no intercept)" begin
        comp = build_formula_components(@formula(y ~ 0 + AR1(time)), data; family = Normal)
        @test size(comp.A) == (n, n)
        @test nnz(comp.A) == n
        ks = Set(keys(comp.hyperparams))
        @test (:τ_ar1 in ks) && (:ρ_ar1 in ks)
        @test length(comp.combined_model) == n
    end

    @testset "AR1 + Intercept (fixed effects)" begin
        comp = build_formula_components(@formula(y ~ 1 + AR1(time)), data; family = Normal)
        @test size(comp.A) == (n, n + 1)
        @test comp.meta.n_fixed == 1
        @test length(comp.combined_model) == n + 1
    end

    @testset "Binomial observations wrapping" begin
        dataB = (
            y = [3, 1, 4],
            n = [5, 6, 7],
            group = [1, 2, 1],
        )
        comp = build_formula_components(@formula(y ~ 0 + IID(group)), dataB; family = Binomial, trials = :n)
        @test comp.y isa GaussianMarkovRandomFields.BinomialObservations
        @test length(comp.y) == length(dataB.y)
        @test size(comp.A, 2) == length(unique(dataB.group))
    end

    @testset "Besag (functor carries W)" begin
        # Simple 3-node chain adjacency
        W = spzeros(3, 3)
        W[1, 2] = 1; W[2, 1] = 1
        W[2, 3] = 1; W[3, 2] = 1

        dataB = (
            y = randn(6),
            region = [1, 2, 3, 2, 1, 3],
        )
        besag = Besag(W)
        comp = build_formula_components(@formula(y ~ 0 + besag(region)), dataB; family = Normal)
        @test size(comp.A) == (length(dataB.y), 3)
        @test nnz(comp.A) == length(dataB.y)
        ks = Set(keys(comp.hyperparams))
        @test :τ_besag in ks
    end
end
