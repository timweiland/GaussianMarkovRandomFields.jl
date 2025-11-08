# COV_EXCL_START
# Random-effect term base and mapping helpers

abstract type FormulaRandomEffectTerm <: StatsModels.AbstractTerm end

# IID(group)
struct IIDTerm <: FormulaRandomEffectTerm
    variable::Symbol
    constraint::Union{Nothing, Symbol, Tuple{AbstractMatrix, AbstractVector}}
end

# IID instance (e.g., iid = IID(); @formula(y ~ iid(group)) or iid_sz = IID(constraint=:sumtozero); @formula(y ~ iid_sz(group)))
function StatsModels.apply_schema(
        t::StatsModels.FunctionTerm{<:GaussianMarkovRandomFields.IID},
        ::StatsModels.Schema,
        ::Type
    )
    var_term = only(t.args)
    return IIDTerm(var_term.sym, t.f.constraint)
end

StatsModels.termvars(term::IIDTerm) = [term.variable]

# RandomWalk(index) → order stored in functor
struct RandomWalkTerm{Order} <: FormulaRandomEffectTerm
    variable::Symbol
    additional_constraints::Union{Nothing, Tuple{AbstractMatrix, AbstractVector}}
end

# RandomWalk instance (e.g., rw1 = RandomWalk(); @formula(y ~ rw1(time)) or rw1 = RandomWalk(1); @formula(y ~ rw1(time)))
function StatsModels.apply_schema(
        t::StatsModels.FunctionTerm{<:GaussianMarkovRandomFields.RandomWalk},
        ::StatsModels.Schema,
        ::Type
    )
    var_term = only(t.args)
    order = t.f.order
    order isa Integer || error("RandomWalk order must be an integer, got $(typeof(order))")
    return RandomWalkTerm{Int(order)}(var_term.sym, t.f.additional_constraints)
end

StatsModels.termvars(term::RandomWalkTerm) = [term.variable]

# AR1(index)
struct AR1Term <: FormulaRandomEffectTerm
    variable::Symbol
    constraint::Union{Nothing, Symbol, Tuple{AbstractMatrix, AbstractVector}}
end

# AR1 instance (e.g., ar1 = AR1(); @formula(y ~ ar1(time)) or ar1_sz = AR1(constraint=:sumtozero); @formula(y ~ ar1_sz(time)))
function StatsModels.apply_schema(
        t::StatsModels.FunctionTerm{<:GaussianMarkovRandomFields.AR1},
        ::StatsModels.Schema,
        ::Type
    )
    var_term = only(t.args)
    return AR1Term(var_term.sym, t.f.constraint)
end

StatsModels.termvars(term::AR1Term) = [term.variable]

# Besag(region; W = adjacency)
struct BesagTerm{M <: AbstractMatrix, MT} <: FormulaRandomEffectTerm
    variable::Symbol
    adjacency::M
    id_to_node::MT
    normalize_var::Bool
    singleton_policy::Symbol
end

function StatsModels.apply_schema(
        t::StatsModels.FunctionTerm{<:GaussianMarkovRandomFields.Besag},
        ::StatsModels.Schema,
        ::Type
    )
    # Functor carries adjacency matrix in t.f.W
    var_term = only(t.args)
    W = t.f.W
    idmap = t.f.id_to_node
    return BesagTerm(var_term.sym, W, idmap, t.f.normalize_var, t.f.singleton_policy)
end

StatsModels.termvars(term::BesagTerm) = [term.variable]

# BYM2(region; W = adjacency)
struct BYM2Term{M <: AbstractMatrix, MT} <: FormulaRandomEffectTerm
    variable::Symbol
    adjacency::M
    id_to_node::MT
    normalize_var::Bool
    singleton_policy::Symbol
    additional_constraints::Union{Nothing, Tuple{AbstractMatrix, AbstractVector}}
    iid_constraint::Union{Nothing, Symbol, Tuple{AbstractMatrix, AbstractVector}}
end

function StatsModels.apply_schema(
        t::StatsModels.FunctionTerm{<:GaussianMarkovRandomFields.BYM2},
        ::StatsModels.Schema,
        ::Type
    )
    # Functor carries adjacency matrix in t.f.W
    var_term = only(t.args)
    W = t.f.W
    idmap = t.f.id_to_node
    return BYM2Term(var_term.sym, W, idmap, t.f.normalize_var, t.f.singleton_policy, t.f.additional_constraints, t.f.iid_constraint)
end

StatsModels.termvars(term::BYM2Term) = [term.variable]

# Sparse mapping helpers
_getcolumn(data, sym::Symbol) = hasproperty(data, sym) ? getproperty(data, sym) : haskey(data, sym) ? data[sym] : error("Variable $(sym) not found in data")

function _levels_and_index(vec)
    # Deterministic ordering: sorted unique
    lvls = sort(collect(unique(vec)))
    idx = Dict{eltype(lvls), Int}()
    for (j, v) in enumerate(lvls)
        idx[v] = j
    end
    return lvls, idx
end

function _indicator_mapping(vec)
    n = length(vec)
    lvls, idx = _levels_and_index(vec)
    m = length(lvls)
    I = Vector{Int}(undef, n)
    J = Vector{Int}(undef, n)
    V = ones(Float64, n)
    @inbounds for i in 1:n
        I[i] = i
        J[i] = idx[vec[i]]
    end
    return sparse(I, J, V, n, m)
end

# StatsModels.modelcols for random terms → return sparse mapping blocks
function StatsModels.modelcols(term::IIDTerm, data)
    v = _getcolumn(data, term.variable)
    return _indicator_mapping(v)
end

function StatsModels.modelcols(term::RandomWalkTerm{1}, data)
    v = _getcolumn(data, term.variable)
    return _indicator_mapping(v)
end

function StatsModels.modelcols(term::AR1Term, data)
    v = _getcolumn(data, term.variable)
    return _indicator_mapping(v)
end

function StatsModels.modelcols(term::BesagTerm, data)
    raw = _getcolumn(data, term.variable)
    n_obs = length(raw)
    n_nodes = size(term.adjacency, 1)

    # Resolve node indices J from raw IDs
    J = Vector{Int}(undef, n_obs)
    if term.id_to_node !== nothing
        idmap = term.id_to_node
        @inbounds for i in 1:n_obs
            id = raw[i]
            idx = try
                idmap[id]
            catch err
                throw(ArgumentError("id_to_node has no entry for $(repr(id)) at row $(i)"))
            end
            idx isa Integer || throw(ArgumentError("id_to_node must map to 1-based integer node indices; got $(typeof(idx))"))
            1 <= idx <= n_nodes || throw(ArgumentError("Mapped node index $(idx) out of bounds 1:$(n_nodes)"))
            J[i] = Int(idx)
        end
    else
        # Expect integers already
        v = Int.(raw)
        @inbounds for i in 1:n_obs
            idx = v[i]
            1 <= idx <= n_nodes || throw(ArgumentError("Region index $(idx) out of bounds 1:$(n_nodes) at row $(i)"))
            J[i] = idx
        end
    end

    I = collect(1:n_obs)
    V = ones(Float64, n_obs)
    return sparse(I, J, V, n_obs, n_nodes)
end

function StatsModels.modelcols(term::BYM2Term, data)
    raw = _getcolumn(data, term.variable)
    n_obs = length(raw)
    n_nodes = size(term.adjacency, 1)

    # Resolve node indices J from raw IDs (same as Besag)
    J = Vector{Int}(undef, n_obs)
    if term.id_to_node !== nothing
        idmap = term.id_to_node
        @inbounds for i in 1:n_obs
            id = raw[i]
            idx = try
                idmap[id]
            catch err
                throw(ArgumentError("id_to_node has no entry for $(repr(id)) at row $(i)"))
            end
            idx isa Integer || throw(ArgumentError("id_to_node must map to 1-based integer node indices; got $(typeof(idx))"))
            1 <= idx <= n_nodes || throw(ArgumentError("Mapped node index $(idx) out of bounds 1:$(n_nodes)"))
            J[i] = Int(idx)
        end
    else
        # Expect integers already
        v = Int.(raw)
        @inbounds for i in 1:n_obs
            idx = v[i]
            1 <= idx <= n_nodes || throw(ArgumentError("Region index $(idx) out of bounds 1:$(n_nodes) at row $(i)"))
            J[i] = idx
        end
    end

    # BYM2 model has 2n components: [u* (spatial); v* (unstructured)]
    # Each observation maps to both: spatial component j and unstructured component n+j
    I_combined = vcat(collect(1:n_obs), collect(1:n_obs))
    J_combined = vcat(J, J .+ n_nodes)  # First half maps to 1:n, second half to (n+1):2n
    V_combined = ones(Float64, 2 * n_obs)

    return sparse(I_combined, J_combined, V_combined, n_obs, 2 * n_nodes)
end

# Separable(component1, component2, ..., componentN)
struct SeparableTerm{T <: Tuple} <: FormulaRandomEffectTerm
    component_terms::T  # Stores the actual term objects (IIDTerm, AR1Term, etc.)
end

# Separable instance (e.g., st = Separable(RW1(), Besag(W)); @formula(y ~ st(time, region)))
function StatsModels.apply_schema(
        t::StatsModels.FunctionTerm{<:GaussianMarkovRandomFields.Separable},
        sch::StatsModels.Schema,
        Typ::Type
    )
    # Extract variable terms and component functors
    var_terms = t.args
    component_functors = t.f.components

    # Validate: length(variables) == length(component_functors)
    length(var_terms) == length(component_functors) ||
        error("Number of variables ($(length(var_terms))) must match number of components ($(length(component_functors)))")

    # For each component, create a proper FunctionTerm and apply schema to it
    # This properly leverages StatsModels' existing apply_schema methods for each functor type
    component_terms = Tuple(
        StatsModels.apply_schema(
                # Create a FunctionTerm(functor, [var_term], expr)
                # The expr is a dummy since we're constructing this synthetically
                StatsModels.FunctionTerm(functor, [var_term], :($(Symbol("dummy_", i))($(var_term.sym)))),
                sch,
                Typ
            )
            for (i, (functor, var_term)) in enumerate(zip(component_functors, var_terms))
    )

    return SeparableTerm(component_terms)
end

StatsModels.termvars(term::SeparableTerm) = vcat([StatsModels.termvars(t) for t in term.component_terms]...)

# Row-wise Kronecker (Khatri-Rao) product for separable indicator mapping
function _khatri_rao(A::AbstractMatrix, B::AbstractMatrix)
    size(A, 1) == size(B, 1) || error("Matrices must have same number of rows")
    n = size(A, 1)
    p, q = size(A, 2), size(B, 2)

    # For sparse matrices, construct via IJV
    if A isa SparseMatrixCSC && B isa SparseMatrixCSC
        I_out = Int[]
        J_out = Int[]
        V_out = Float64[]

        for i in 1:n
            # Get nonzeros in row i of A and B
            A_row_nz = [(j, A[i, j]) for j in 1:p if A[i, j] != 0]
            B_row_nz = [(k, B[i, k]) for k in 1:q if B[i, k] != 0]

            # Kronecker product of the two rows
            for (j_A, v_A) in A_row_nz
                for (j_B, v_B) in B_row_nz
                    # Column index in result: (j_A - 1) * q + j_B
                    j_out = (j_A - 1) * q + j_B
                    push!(I_out, i)
                    push!(J_out, j_out)
                    push!(V_out, v_A * v_B)
                end
            end
        end

        return sparse(I_out, J_out, V_out, n, p * q)
    else
        # Fallback for dense matrices
        C = zeros(n, p * q)
        for i in 1:n
            C[i, :] = kron(A[i, :], B[i, :])
        end
        return C
    end
end

# Kronecker indicator mapping for separable models
function StatsModels.modelcols(term::SeparableTerm, data)
    # Get indicator matrix for each component using its modelcols method
    indicators = [StatsModels.modelcols(comp_term, data) for comp_term in term.component_terms]

    # Result is row-wise Kronecker product (Khatri-Rao) of all indicators
    # For SeparableTerm([term1, term2]), we compute: KhatriRao(I1, I2)
    # This gives n_obs × (n_1 * n_2 * ... * n_N) with rightmost component varying fastest
    # Each row is the Kronecker product of the corresponding rows from each indicator
    return foldl(_khatri_rao, indicators)
end

# COV_EXCL_STOP
