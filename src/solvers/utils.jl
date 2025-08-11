using LinearSolve

export ensure_factorization!

"""
    ensure_factorization!(linsolve)

Ensure that the LinearSolve cache has computed its factorization.
This is needed before accessing cached factorization data.
"""
function ensure_factorization!(linsolve)
    if linsolve.isfresh
        LinearSolve.solve!(linsolve)
    end
    return nothing
end