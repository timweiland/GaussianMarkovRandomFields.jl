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

    # Create functor instances
    iid = IID()
    rw1 = RandomWalk()
    ar1 = AR1()

    @testset "IID + RW1 (no intercept)" begin
        comp = build_formula_components(@formula(y ~ 0 + iid(group) + rw1(1, time)), data; family = Normal)
        @test size(comp.A) == (n, 2 + n)
        @test nnz(comp.A) == n + n  # one indicator per block per row
        @test comp.meta.n_random == 2
        @test comp.meta.n_fixed == 0
        @test length(comp.combined_model) == 2 + n
        ks = Set(keys(comp.hyperparams))
        @test (:τ_iid in ks) && (:τ_rw1 in ks)
    end

    @testset "AR1 (no intercept)" begin
        comp = build_formula_components(@formula(y ~ 0 + ar1(time)), data; family = Normal)
        @test size(comp.A) == (n, n)
        @test nnz(comp.A) == n
        ks = Set(keys(comp.hyperparams))
        @test (:τ_ar1 in ks) && (:ρ_ar1 in ks)
        @test length(comp.combined_model) == n
    end

    @testset "AR1 + Intercept (fixed effects)" begin
        comp = build_formula_components(@formula(y ~ 1 + ar1(time)), data; family = Normal)
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
        comp = build_formula_components(@formula(y ~ 0 + iid(group)), dataB; family = Binomial, trials = :n)
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

    @testset "Besag with id_to_node mapping" begin
        # 3-node chain adjacency
        W = spzeros(3, 3)
        W[1, 2] = 1; W[2, 1] = 1
        W[2, 3] = 1; W[3, 2] = 1

        # Non-integer region IDs
        names = ["A", "B", "C", "B", "A", "C"]
        dataC = (
            y = randn(length(names)),
            region_name = names,
        )
        idmap = Dict("A" => 1, "B" => 2, "C" => 3)
        besag = Besag(W; id_to_node = idmap)
        comp = build_formula_components(@formula(y ~ 0 + besag(region_name)), dataC; family = Normal)
        @test size(comp.A) == (length(names), 3)
        @test nnz(comp.A) == length(names)
        # Check mapping: rows where region_name == "C" should map to column 3
        rows_C = findall(==("C"), names)
        for r in rows_C
            @test comp.A[r, 3] == 1.0
        end
        ks = Set(keys(comp.hyperparams))
        @test :τ_besag in ks
    end
end

@testset "Formula Interface with Constraints" begin
    # Common test data
    n = 20
    data = (
        y = randn(n),
        group = repeat([1, 2, 3, 4], 5),
        time = collect(1:n),
    )

    @testset "IID with sum-to-zero constraint" begin
        iid_sz = IID(constraint = :sumtozero)
        comp = build_formula_components(@formula(y ~ 0 + iid_sz(group)), data; family = Normal)

        # Should create ConstrainedGMRF
        gmrf = comp.combined_model(τ_iid = 1.0)
        @test gmrf isa ConstrainedGMRF

        # Verify sum-to-zero constraint is satisfied
        samples = [rand(gmrf) for _ in 1:5]
        for s in samples
            @test abs(sum(s)) < 1.0e-10
        end
    end

    @testset "AR1 with sum-to-zero constraint" begin
        ar1_sz = AR1(constraint = :sumtozero)
        comp = build_formula_components(@formula(y ~ 0 + ar1_sz(time)), data; family = Normal)

        # Should create ConstrainedGMRF
        gmrf = comp.combined_model(τ_ar1 = 1.0, ρ_ar1 = 0.7)
        @test gmrf isa ConstrainedGMRF

        # Verify sum-to-zero constraint is satisfied
        samples = [rand(gmrf) for _ in 1:5]
        for s in samples
            @test abs(sum(s)) < 1.0e-10
        end
    end

    @testset "RandomWalk with built-in sum-to-zero" begin
        rw1 = RandomWalk()
        comp = build_formula_components(@formula(y ~ 0 + rw1(1, time)), data; family = Normal)

        # RW1 always has built-in sum-to-zero constraint
        gmrf = comp.combined_model(τ_rw1 = 1.0)
        @test gmrf isa ConstrainedGMRF

        # Verify sum-to-zero constraint is satisfied
        samples = [rand(gmrf) for _ in 1:5]
        for s in samples
            @test abs(sum(s)) < 1.0e-10
        end
    end

    @testset "IID with custom constraint" begin
        # Custom constraint: x1 + x2 = 0
        A_custom = zeros(1, 4)
        A_custom[1, 1] = 1.0
        A_custom[1, 2] = 1.0
        e_custom = [0.0]

        iid_custom = IID(constraint = (A_custom, e_custom))
        comp = build_formula_components(@formula(y ~ 0 + iid_custom(group)), data; family = Normal)

        gmrf = comp.combined_model(τ_iid = 1.0)
        @test gmrf isa ConstrainedGMRF

        # Verify custom constraint is satisfied
        samples = [rand(gmrf) for _ in 1:5]
        for s in samples
            @test abs(s[1] + s[2]) < 1.0e-10
        end
    end

    @testset "Unconstrained models remain unconstrained" begin
        iid = IID()
        ar1 = AR1()

        comp_iid = build_formula_components(@formula(y ~ 0 + iid(group)), data; family = Normal)
        gmrf_iid = comp_iid.combined_model(τ_iid = 1.0)
        @test !(gmrf_iid isa ConstrainedGMRF)

        comp_ar1 = build_formula_components(@formula(y ~ 0 + ar1(time)), data; family = Normal)
        gmrf_ar1 = comp_ar1.combined_model(τ_ar1 = 1.0, ρ_ar1 = 0.5)
        @test !(gmrf_ar1 isa ConstrainedGMRF)
    end

    @testset "Combined model with multiple constrained terms" begin
        iid_sz = IID(constraint = :sumtozero)
        rw1 = RandomWalk()

        comp = build_formula_components(@formula(y ~ 0 + iid_sz(group) + rw1(1, time)), data; family = Normal)
        gmrf = comp.combined_model(τ_iid = 1.0, τ_rw1 = 1.0)

        # Combined model should be constrained
        @test gmrf isa ConstrainedGMRF

        # Verify dimensions
        @test length(gmrf) == 4 + n  # 4 groups + n time points

        # Sample and check structure
        s = rand(gmrf)
        @test length(s) == 4 + n
    end

    @testset "Constraint dimension validation" begin
        # Wrong dimension constraint matrix should error
        A_wrong = ones(1, 5)  # 5 columns, but only 4 groups
        e_wrong = [0.0]
        iid_wrong = IID(constraint = (A_wrong, e_wrong))

        @test_throws ErrorException build_formula_components(@formula(y ~ 0 + iid_wrong(group)), data; family = Normal)
    end
end
