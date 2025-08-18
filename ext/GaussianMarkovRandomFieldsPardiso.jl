# COV_EXCL_START
module GaussianMarkovRandomFieldsPardiso

using GaussianMarkovRandomFields
using Pardiso
using LinearSolve
using LinearAlgebra

# Selected inversion implementations for LinearSolve.jl integration
function GaussianMarkovRandomFields._selinv_diag_impl(linsolve, ::LinearSolve.PardisoJL)
    # Access PardisoSolver through cacheval
    ps = LinearSolve.@get_cacheval(linsolve, :PardisoJL)

    # Set up for selected inversion
    Pardiso.set_phase!(ps, Pardiso.SELECTED_INVERSION)
    Pardiso.set_iparm!(ps, 36, 1)  # allocate new memory for selected inverse

    # Create output matrix (will contain selected inverse)
    B = similar(linsolve.A)
    x = Array{Float64}(undef, size(linsolve.A, 1))

    # Perform selected inversion
    Pardiso.pardiso(ps, x, B, x)

    # Reset to solve phase for future operations
    Pardiso.set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)

    return diag(B)
end

function GaussianMarkovRandomFields._selinv_impl(linsolve, ::LinearSolve.PardisoJL)
    # Access PardisoSolver through cacheval
    ps = LinearSolve.@get_cacheval(linsolve, :PardisoJL)

    # Set up for selected inversion
    Pardiso.set_phase!(ps, Pardiso.SELECTED_INVERSION)
    Pardiso.set_iparm!(ps, 36, 1)  # allocate new memory for selected inverse

    # Create output matrix
    B = similar(linsolve.A)
    x = Array{Float64}(undef, size(linsolve.A, 1))

    # Perform selected inversion
    Pardiso.pardiso(ps, x, B, x)

    # Reset to solve phase for future operations
    Pardiso.set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)

    return B
end

# Log determinant implementation for LinearSolve.jl integration
function GaussianMarkovRandomFields._logdet_cov_impl(linsolve, ::LinearSolve.PardisoJL)
    # Access PardisoSolver through cacheval
    ps = LinearSolve.@get_cacheval(linsolve, :PardisoJL)

    # The log determinant of the precision matrix is stored in dparm[33]
    # We want log det(Σ) = log det(Q^-1) = -log det(Q)
    return -ps.dparm[33]
end

# Pardiso doesn't support backward solves due to LDL^T factorization complexity
GaussianMarkovRandomFields.supports_backward_solve(::LinearSolve.PardisoJL) = Val{false}()

# Configure Pardiso with optimal defaults for GMRF operations
function GaussianMarkovRandomFields.configure_algorithm(alg::LinearSolve.PardisoJL)
    # Set default matrix type if not specified
    matrix_type = alg.matrix_type === nothing ? Pardiso.REAL_SYM_INDEF : alg.matrix_type

    # Set default iparm for log determinant computation if not specified
    iparm = if alg.iparm === nothing
        [(33, 1)]  # Enable log determinant computation
    else
        # Check if iparm[33] is already set, if not add it
        has_logdet = any(pair -> pair[1] == 33, alg.iparm)
        has_logdet ? alg.iparm : [alg.iparm..., (33, 1)]
    end

    return LinearSolve.PardisoJL(matrix_type = matrix_type, iparm = iparm)
end

end # module
# COV_EXCL_STOP
