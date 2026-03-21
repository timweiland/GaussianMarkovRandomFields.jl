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

@testset "Formula Interface - Matern SPDE" begin
    # Generate random 2D observation points
    n_pts = 20
    x_coords = randn(n_pts)
    y_coords = randn(n_pts)
    data_spde = (
        y = randn(n_pts),
        x_coord = x_coords,
        y_coord = y_coords,
    )

    @testset "Matern auto-mesh (no intercept)" begin
        matern = Matern(smoothness = 1)
        comp = build_formula_components(@formula(y ~ 0 + matern(x_coord, y_coord)), data_spde; family = Normal)

        # Design matrix: n_obs × ndofs (mesh DOFs >= n_obs for convex hull)
        @test size(comp.A, 1) == n_pts
        @test size(comp.A, 2) > 0
        @test comp.meta.n_random == 1
        @test comp.meta.n_fixed == 0

        # Hyperparameters should include τ and range
        ks = Set(keys(comp.hyperparams))
        @test :τ_matern in ks
        @test :range_matern in ks

        # Latent dimension must match design matrix columns
        @test length(comp.combined_model) == size(comp.A, 2)

        # Should produce a valid GMRF
        gmrf = comp.combined_model(τ_matern = 1.0, range_matern = 1.0)
        @test length(gmrf) == size(comp.A, 2)
    end

    @testset "Matern with intercept" begin
        matern = Matern(smoothness = 1)
        comp = build_formula_components(@formula(y ~ 1 + matern(x_coord, y_coord)), data_spde; family = Normal)

        @test size(comp.A, 1) == n_pts
        @test comp.meta.n_random == 1
        @test comp.meta.n_fixed == 1
        @test length(comp.combined_model) == size(comp.A, 2)
    end

    @testset "Matern with pre-built discretization" begin
        # Build a MaternModel to extract its discretization
        points = hcat(x_coords, y_coords)
        ref_model = MaternModel(points; smoothness = 1)
        disc = ref_model.discretization

        matern = Matern(disc; smoothness = 1)
        comp = build_formula_components(@formula(y ~ 0 + matern(x_coord, y_coord)), data_spde; family = Normal)

        @test size(comp.A) == (n_pts, length(ref_model))
        @test length(comp.combined_model) == length(ref_model)
    end

    @testset "Matern with smoothness=2" begin
        matern = Matern(smoothness = 2)
        comp = build_formula_components(@formula(y ~ 0 + matern(x_coord, y_coord)), data_spde; family = Normal)

        @test size(comp.A, 1) == n_pts
        ks = Set(keys(comp.hyperparams))
        @test :τ_matern in ks
        @test :range_matern in ks
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

    @testset "RW2 via formula interface" begin
        rw2 = RandomWalk(2)
        comp = build_formula_components(@formula(y ~ 0 + rw2(time)), data; family = Normal)

        # Design matrix: n observations mapping to n time points
        @test size(comp.A) == (n, n)
        @test nnz(comp.A) == n

        # Hyperparameters should use rw2 suffix
        ks = Set(keys(comp.hyperparams))
        @test :τ_rw2 in ks

        # Should produce ConstrainedGMRF (intrinsic with 2 constraints)
        gmrf = comp.combined_model(τ_rw2 = 1.0)
        @test gmrf isa ConstrainedGMRF
        @test length(gmrf) == n

        # Sampling should work
        s = rand(gmrf)
        @test length(s) == n
    end

    @testset "AR(2) via formula interface" begin
        ar2 = AR(2)
        comp = build_formula_components(@formula(y ~ 0 + ar2(time)), data; family = Normal)

        @test size(comp.A) == (n, n)
        @test nnz(comp.A) == n

        # Hyperparameters should use ar2 suffix
        ks = Set(keys(comp.hyperparams))
        @test :τ_ar2 in ks
        @test :pacf1_ar2 in ks
        @test :pacf2_ar2 in ks

        # Should produce unconstrained GMRF
        gmrf = comp.combined_model(τ_ar2 = 1.0, pacf1_ar2 = 0.5, pacf2_ar2 = -0.3)
        @test gmrf isa GMRF
        @test length(gmrf) == n

        # Sampling should work
        s = rand(gmrf)
        @test length(s) == n
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

@testset "predict_cols" begin
    # Helper: extract random effect terms from a formula applied to data
    function _random_terms(formula, data)
        schema = StatsModels.schema(formula, data)
        tf = StatsModels.apply_schema(formula, schema)
        rhs = tf.rhs isa StatsModels.MatrixTerm ? tf.rhs.terms : [tf.rhs]
        # Filter out standard StatsModels terms (InterceptTerm, ContinuousTerm, etc.)
        return [t for t in rhs if !(t isa Union{StatsModels.InterceptTerm, StatsModels.ContinuousTerm, StatsModels.CategoricalTerm})]
    end

    @testset "IID / RW / AR1 — column count matches model" begin
        n = 10
        train = (y = randn(n), group = repeat(1:5, 2), time = collect(1:n))

        iid = IID()
        rw1 = RandomWalk()
        ar1 = AR1()

        f = @formula(y ~ 0 + iid(group) + rw1(time) + ar1(time))
        comp = build_formula_components(f, train; family = Normal)
        terms = _random_terms(f, train)
        cm = comp.combined_model

        # Prediction with subset of levels — columns must still match model dim
        test_subset = (group = [2, 1], time = [3, 5])
        for (term, model) in zip(terms, [cm.iid, cm.rw1, cm.ar1])
            A_pred = predict_cols(term, model, test_subset)
            @test size(A_pred, 1) == 2
            @test size(A_pred, 2) == length(model)
        end

        # Non-contiguous levels: column indices use training level mapping
        test_gap = (group = [1, 3, 5], time = [1, 5, 10])
        iid_term = terms[1]
        A_gap = predict_cols(iid_term, cm.iid, test_gap)
        @test size(A_gap) == (3, length(cm.iid))
        @test A_gap[1, 1] == 1.0  # group 1 → col 1
        @test A_gap[2, 3] == 1.0  # group 3 → col 3
        @test A_gap[3, 5] == 1.0  # group 5 → col 5

        # Prediction with all levels — should also work
        test_all = (group = [1, 2, 3, 4, 5], time = collect(1:n))
        for (term, model) in zip(terms, [cm.iid, cm.rw1, cm.ar1])
            A_pred = predict_cols(term, model, test_all)
            @test size(A_pred, 2) == length(model)
        end

        # Unseen level should error
        @test_throws ArgumentError predict_cols(iid_term, cm.iid, (group = [99],))
    end

    @testset "Non-1-indexed levels (e.g. years)" begin
        years = repeat(1900:1904, 2)
        train = (y = randn(10), year = years)
        rw1 = RandomWalk()

        f = @formula(y ~ 0 + rw1(year))
        comp = build_formula_components(f, train; family = Normal)
        rw_model = comp.combined_model.rw1
        term = _random_terms(f, train)[1]

        # Model should have 5 levels (1900..1904)
        @test length(rw_model) == 5
        @test rw_model.levels == collect(1900:1904)

        # Predict at subset — year 1902 should map to column 3
        test = (year = [1902, 1904],)
        A_pred = predict_cols(term, rw_model, test)
        @test size(A_pred) == (2, 5)
        @test A_pred[1, 3] == 1.0  # 1902 → col 3
        @test A_pred[2, 5] == 1.0  # 1904 → col 5

        # Unseen year should error
        @test_throws ArgumentError predict_cols(term, rw_model, (year = [2000],))
    end

    @testset "RW with gaps fills unobserved nodes" begin
        # Training data has years 1900, 1901, 1903, 1905 (gaps at 1902, 1904)
        train = (y = randn(8), year = repeat([1900, 1901, 1903, 1905], 2))
        rw1 = RandomWalk()

        f = @formula(y ~ 0 + rw1(year))
        comp = build_formula_components(f, train; family = Normal)
        rw_model = comp.combined_model.rw1
        term = _random_terms(f, train)[1]

        # Model should span full range including gaps
        @test length(rw_model) == 6  # 1900:1905
        @test rw_model.levels == collect(1900:1905)

        # Design matrix should have 6 columns, not 4
        @test size(comp.A, 2) == 6

        # Year 1900 → col 1, year 1903 → col 4 (not col 3!)
        A = comp.A
        @test A[1, 1] == 1.0   # first obs is year 1900 → col 1
        @test A[3, 4] == 1.0   # third obs is year 1903 → col 4

        # Can predict at gap year 1902
        test = (year = [1902],)
        A_pred = predict_cols(term, rw_model, test)
        @test size(A_pred) == (1, 6)
        @test A_pred[1, 3] == 1.0  # 1902 → col 3
    end

    @testset "RW/AR require integer indices" begin
        rw1 = RandomWalk()
        ar1 = AR1()

        # Float indices should error
        train_float = (y = randn(3), time = [1.0, 2.0, 3.0])
        @test_throws ArgumentError build_formula_components(@formula(y ~ 0 + rw1(time)), train_float; family = Normal)
        @test_throws ArgumentError build_formula_components(@formula(y ~ 0 + ar1(time)), train_float; family = Normal)

        # String indices should error
        train_str = (y = randn(3), time = ["a", "b", "c"])
        @test_throws ArgumentError build_formula_components(@formula(y ~ 0 + rw1(time)), train_str; family = Normal)
    end

    @testset "Besag — same as modelcols" begin
        W = spzeros(3, 3)
        W[1, 2] = 1; W[2, 1] = 1; W[2, 3] = 1; W[3, 2] = 1
        besag = Besag(W)

        train = (y = randn(6), region = [1, 2, 3, 2, 1, 3])
        test = (region = [3, 1],)

        f = @formula(y ~ 0 + besag(region))
        comp = build_formula_components(f, train; family = Normal)
        term = _random_terms(f, train)[1]

        A_pred = predict_cols(term, comp.combined_model.besag, test)
        A_mc = StatsModels.modelcols(term, test)
        @test A_pred == A_mc
    end

    @testset "Matern — reuses discretization" begin
        x_train = randn(20)
        y_train = randn(20)
        train = (y = randn(20), x_coord = x_train, y_coord = y_train)

        x_test = randn(5)
        y_test = randn(5)
        test = (x_coord = x_test, y_coord = y_test)

        matern = Matern(smoothness = 1)
        f = @formula(y ~ 0 + matern(x_coord, y_coord))
        comp = build_formula_components(f, train; family = Normal)
        matern_model = comp.combined_model.matern
        term = _random_terms(f, train)[1]

        A_pred = predict_cols(term, matern_model, test)

        # Should have n_test rows and same columns as model dimension
        @test size(A_pred, 1) == 5
        @test size(A_pred, 2) == length(matern_model)

        # Should match direct evaluation_matrix call
        pts = hcat(Float64.(x_test), Float64.(y_test))
        A_direct = evaluation_matrix(matern_model, pts)
        @test A_pred == A_direct
    end

    @testset "Separable — recursive predict_cols" begin
        n_time = 5
        n_space = 3
        n_obs = n_time * n_space

        W_space = spzeros(n_space, n_space)
        for i in 1:(n_space - 1)
            W_space[i, i + 1] = 1
            W_space[i + 1, i] = 1
        end

        train = (
            y = randn(n_obs),
            time = repeat(1:n_time, outer = n_space),
            space = repeat(1:n_space, inner = n_time),
        )
        test = (time = [2, 4], space = [1, 3])

        rw1 = RandomWalk()
        besag = Besag(W_space)
        st = Separable(rw1, besag)

        f = @formula(y ~ 0 + st(time, space))
        comp = build_formula_components(f, train; family = Normal)
        sep_model = comp.combined_model.separable
        term = _random_terms(f, train)[1]

        A_pred = predict_cols(term, sep_model, test)
        @test size(A_pred, 1) == 2
        @test size(A_pred, 2) == length(sep_model)
    end
end
