using ReTest
using StatsModels
using Distributions
using GaussianMarkovRandomFields
using SparseArrays

@testset "Formula Interface" begin
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
    rw1 = RandomWalk(1)
    ar1 = AR1()

    @testset "IID + RW1 (no intercept)" begin
        comp = build_formula_components(@formula(y ~ 0 + iid(group) + rw1(time)), data; family = Normal)
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
        rw1 = RandomWalk(1)
        comp = build_formula_components(@formula(y ~ 0 + rw1(time)), data; family = Normal)

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
        rw1 = RandomWalk(1)

        comp = build_formula_components(@formula(y ~ 0 + iid_sz(group) + rw1(time)), data; family = Normal)
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

    @testset "Separable (Kronecker product) models" begin
        # Create space-time data
        n_time = 10
        n_space = 4
        n_obs = n_time * n_space

        data_st = (
            y = randn(n_obs),
            time = repeat(1:n_time, outer = n_space),
            space = repeat(1:n_space, inner = n_time),
        )

        # Simple spatial adjacency (chain)
        W_space = spzeros(n_space, n_space)
        for i in 1:(n_space - 1)
            W_space[i, i + 1] = 1
            W_space[i + 1, i] = 1
        end

        @testset "Basic Separable (RW1 ⊗ Besag)" begin
            rw1 = RandomWalk()
            besag = Besag(W_space)
            st = Separable(rw1, besag)

            comp = build_formula_components(@formula(y ~ 1 + st(time, space)), data_st; family = Normal)

            @test size(comp.A) == (n_obs, n_time * n_space + 1)  # +1 for intercept
            @test comp.meta.n_random == 1
            @test comp.meta.n_fixed == 1
            @test length(comp.combined_model) == n_time * n_space + 1

            # Check hyperparameters
            ks = Set(keys(comp.hyperparams))
            @test :τ_rw1_separable in ks
            @test :τ_besag_separable in ks
        end

        @testset "3-way Separable model" begin
            # Add a third dimension
            n_group = 2
            n_obs_3d = n_obs * n_group

            data_3d = (
                y = randn(n_obs_3d),
                time = repeat(repeat(1:n_time, outer = n_space), outer = n_group),
                space = repeat(repeat(1:n_space, inner = n_time), outer = n_group),
                group = repeat(1:n_group, inner = n_obs),
            )

            rw1 = RandomWalk()
            iid_space = IID()
            iid_group = IID()
            sep3 = Separable(rw1, iid_space, iid_group)

            comp = build_formula_components(@formula(y ~ 0 + sep3(time, space, group)), data_3d; family = Normal)

            @test size(comp.A, 1) == n_obs_3d
            @test size(comp.A, 2) == n_time * n_space * n_group

            # Hyperparameters for all three components
            ks = Set(keys(comp.hyperparams))
            @test :τ_rw1_separable in ks
            @test :τ_iid_separable in ks
            @test :τ_iid_2_separable in ks
        end

        @testset "Separable GMRF instantiation" begin
            rw1 = RandomWalk()
            besag = Besag(W_space)
            st = Separable(rw1, besag)

            comp = build_formula_components(@formula(y ~ 0 + st(time, space)), data_st; family = Normal)

            # Should create GMRF successfully with constraints
            gmrf = comp.combined_model(τ_rw1_separable = 1.0, τ_besag_separable = 1.0)
            @test length(gmrf) == n_time * n_space

            # Different hyperparameter values should give different precision matrices
            gmrf2 = comp.combined_model(τ_rw1_separable = 2.0, τ_besag_separable = 1.0)
            Q1 = Matrix(precision_map(gmrf))
            Q2 = Matrix(precision_map(gmrf2))
            @test !(Q1 ≈ Q2)
        end
    end
end
