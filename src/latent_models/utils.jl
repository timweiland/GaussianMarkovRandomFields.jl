"""
    _process_constraint(constraint, n::Int)

Process user-provided constraint specification into stored format.

# Arguments
- `constraint`: Can be `nothing`, `:sumtozero`, or `(A, e)` tuple
- `n`: Size of the model

# Returns
- `nothing` for no constraints
- `(A, e)` tuple for constraints where `A*x = e`

# Examples
```julia
# No constraints
_process_constraint(nothing, 10)  # returns nothing

# Sum-to-zero constraint
_process_constraint(:sumtozero, 10)  # returns (ones(1, 10), [0.0])

# Custom constraint
A = [1.0 1.0 0.0 ...]
e = [1.0]
_process_constraint((A, e), 10)  # returns (A, e) after validation
```
"""
# No constraints
_process_constraint(::Nothing, n::Int) = nothing

# Sum-to-zero constraint (dispatch on symbol via Val)
_process_constraint(constraint::Symbol, n::Int) = _process_constraint(Val(constraint), n)

function _process_constraint(::Val{:sumtozero}, n::Int)
    A = ones(1, n)
    e = [0.0]
    return (A, e)
end

function _process_constraint(::Val{S}, n::Int) where {S}
    throw(ArgumentError("Unknown constraint symbol :$S. Use :sumtozero or provide (A, e) tuple"))
end

# Custom constraint
function _process_constraint(constraint::Tuple{AbstractMatrix, AbstractVector}, n::Int)
    A, e = constraint
    size(A, 2) == n || throw(ArgumentError("Constraint matrix columns ($(size(A, 2))) must match model size ($n)"))
    size(A, 1) == length(e) || throw(ArgumentError("Constraint matrix rows ($(size(A, 1))) must match constraint vector length ($(length(e)))"))
    return (A, e)
end
