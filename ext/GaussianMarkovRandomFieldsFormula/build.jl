# LatentModel construction and formula components builder

# LatentModel construction per term
function _latent_model(term::IIDTerm, data)
    v = _getcolumn(data, term.variable)
    lvls, _ = _levels_and_index(v)
    n = length(lvls)

    # Validate custom constraint dimensions if provided
    if term.constraint isa Tuple
        A, e = term.constraint
        if size(A, 2) != n
            error("Constraint matrix for $(term.variable) has $(size(A, 2)) columns but variable has $(n) levels")
        end
    end

    return IIDModel(n; constraint = term.constraint)
end

function _latent_model(term::RandomWalkTerm{1}, data)
    v = _getcolumn(data, term.variable)
    lvls, _ = _levels_and_index(v)
    n = length(lvls)

    # Validate custom constraint dimensions if provided
    if term.additional_constraints isa Tuple
        A, e = term.additional_constraints
        if size(A, 2) != n
            error("Additional constraint matrix for $(term.variable) has $(size(A, 2)) columns but variable has $(n) levels")
        end
    end

    return RW1Model(n; additional_constraints = term.additional_constraints)
end

function _latent_model(term::AR1Term, data)
    v = _getcolumn(data, term.variable)
    lvls, _ = _levels_and_index(v)
    n = length(lvls)

    # Validate custom constraint dimensions if provided
    if term.constraint isa Tuple
        A, e = term.constraint
        if size(A, 2) != n
            error("Constraint matrix for $(term.variable) has $(size(A, 2)) columns but variable has $(n) levels")
        end
    end

    return AR1Model(n; constraint = term.constraint)
end

function _latent_model(term::BesagTerm, _)
    # Pass type-stable Val options to BesagModel
    return BesagModel(
        term.adjacency;
        normalize_var = Val(term.normalize_var),
        singleton_policy = Val(term.singleton_policy),
    )
end

function _latent_model(term::BYM2Term, _)
    # BYM2 always uses normalize_var=true
    # Validate additional constraints if provided
    if term.additional_constraints isa Tuple
        A, e = term.additional_constraints
        n = size(term.adjacency, 1)
        if size(A, 2) != n
            error("Additional constraint matrix for BYM2 term has $(size(A, 2)) columns but adjacency has $(n) nodes")
        end
    end

    return BYM2Model(
        term.adjacency;
        normalize_var = Val(term.normalize_var),
        singleton_policy = Val(term.singleton_policy),
        additional_constraints = term.additional_constraints,
    )
end

# Column widths for each term (avoid relying on internal StatsModels widths)
_ncols_for_term(term, data) = size(StatsModels.modelcols(term, data), 2)

# Build components
function GaussianMarkovRandomFields.build_formula_components(
        formula::StatsModels.FormulaTerm,
        data;
        family = Distributions.Normal,
        trials = nothing,
        fixed_prior::Real = 1.0e-6,
    )
    # Transform formula
    schema = StatsModels.schema(formula, data)
    tf = StatsModels.apply_schema(formula, schema)

    # Response
    y = StatsModels.modelcols(tf.lhs, data)
    if family == Distributions.Binomial
        trials === nothing && error("family=Binomial requires trials keyword specifying a column name")
        tcol = _getcolumn(data, trials)
        length(tcol) == length(y) || error("trials length $(length(tcol)) must match response length $(length(y))")
        y = BinomialObservations(y, tcol)
    end

    # Partition terms
    rhs_terms = tf.rhs isa StatsModels.MatrixTerm ? tf.rhs.terms : [tf.rhs]
    random_terms = FormulaRandomEffectTerm[]
    fixed_terms = Any[]
    for t in rhs_terms
        if t isa FormulaRandomEffectTerm
            push!(random_terms, t)
        else
            push!(fixed_terms, t)
        end
    end

    # Build random mapping blocks and models
    A_blocks = Vector{SparseMatrixCSC{Float64, Int}}()
    models = Vector{GaussianMarkovRandomFields.LatentModel}()
    for rt in random_terms
        A_i = StatsModels.modelcols(rt, data)
        A_i isa SparseMatrixCSC || (A_i = sparse(A_i))
        push!(A_blocks, A_i)
        push!(models, _latent_model(rt, data))
    end

    # Build fixed-effects design from fixed terms only
    n_fixed = 0
    if !isempty(fixed_terms)
        fixed_cols = Vector{AbstractMatrix}()
        for ft in fixed_terms
            X = StatsModels.modelcols(ft, data)
            X isa AbstractMatrix || (X = reshape(X, :, 1))
            push!(fixed_cols, X)
        end
        X_fixed = hcat(fixed_cols...)
        X_fixed = sparse(X_fixed)
        n_fixed = size(X_fixed, 2)
        push!(A_blocks, X_fixed)
        if n_fixed > 0
            push!(models, FixedEffectsModel(n_fixed; λ = fixed_prior))
        end
    end

    # Assemble A and CombinedModel
    isempty(A_blocks) && error("No terms found in RHS to construct design matrix")
    A = hcat(A_blocks...)

    isempty(models) && error("No latent components constructed (check formula RHS)")
    combined_model = CombinedModel(models...)

    # Observation model
    base_model = ExponentialFamily(family)
    obs_model = LinearlyTransformedObservationModel(base_model, A)

    # Hyperparameters (prefixed)
    hyperparams = GaussianMarkovRandomFields.hyperparameters(combined_model)

    meta = (
        n_random = length(random_terms),
        n_fixed = n_fixed,
        term_sizes = (
            random = [size(StatsModels.modelcols(t, data), 2) for t in random_terms],
            fixed = [size(StatsModels.modelcols(t, data), 2) for t in fixed_terms],
        ),
    )

    # Sanity: columns of A must match latent dimension
    n_cols = size(A, 2)
    n_lat = length(combined_model)
    n_cols == n_lat || error("Design matrix columns ($n_cols) do not match latent dimension ($n_lat)")

    return (
        A = A,
        y = y,
        obs_model = obs_model,
        combined_model = combined_model,
        hyperparams = hyperparams,
        meta = meta,
    )
end
