# Random-effect term base and mapping helpers

abstract type FormulaRandomEffectTerm <: StatsModels.AbstractTerm end

# IID(group)
struct IIDTerm <: FormulaRandomEffectTerm
    variable::Symbol
end

function StatsModels.apply_schema(
        t::StatsModels.FunctionTerm{typeof(IID)},
        ::StatsModels.Schema,
        ::Type
    )
    var_term = only(t.args)
    return IIDTerm(var_term.sym)
end

StatsModels.termvars(term::IIDTerm) = [term.variable]

# RandomWalk(order, index) → MVP: order == 1
struct RandomWalkTerm{Order} <: FormulaRandomEffectTerm
    variable::Symbol
end

function StatsModels.apply_schema(
        t::StatsModels.FunctionTerm{typeof(RandomWalk)},
        ::StatsModels.Schema,
        ::Type
    )
    order_term, var_term = t.args
    order = order_term isa StatsModels.ConstantTerm ? order_term.n : order_term
    order isa Integer || error("RandomWalk order must be an integer, got $(typeof(order))")
    return RandomWalkTerm{Int(order)}(var_term.sym)
end

StatsModels.termvars(term::RandomWalkTerm) = [term.variable]

# AR1(index)
struct AR1Term <: FormulaRandomEffectTerm
    variable::Symbol
end

function StatsModels.apply_schema(
        t::StatsModels.FunctionTerm{typeof(AR1)},
        ::StatsModels.Schema,
        ::Type
    )
    var_term = only(t.args)
    return AR1Term(var_term.sym)
end

StatsModels.termvars(term::AR1Term) = [term.variable]

# Besag(region; W = adjacency)
struct BesagTerm{M <: AbstractMatrix} <: FormulaRandomEffectTerm
    variable::Symbol
    adjacency::M
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
    return BesagTerm(var_term.sym, W, t.f.normalize_var, t.f.singleton_policy)
end

StatsModels.termvars(term::BesagTerm) = [term.variable]

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
    v = Int.(_getcolumn(data, term.variable))
    n_obs = length(v)
    n_nodes = size(term.adjacency, 1)
    I = collect(1:n_obs)
    J = v
    V = ones(Float64, n_obs)
    return sparse(I, J, V, n_obs, n_nodes)
end
