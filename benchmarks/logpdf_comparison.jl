#!/usr/bin/env julia
"""
Benchmark: Compare GMRF vs ChordalGMRF for logpdf computation

Tests both correctness (results match) and performance on SSMC matrices.

Usage:
    cd benchmarks
    julia --project=. logpdf_comparison.jl
"""

using GaussianMarkovRandomFields
using BenchmarkTools
using Distributions: logpdf
using SparseArrays
using LinearAlgebra
using LinearSolve
using Printf
using Random
using MatrixDepot
using Zygote, Mooncake
using DifferentiationInterface: DifferentiationInterface, AutoZygote, AutoMooncake

println("="^80)
println("LOGPDF COMPARISON: GMRF vs ChordalGMRF")
println("="^80)

# Helper to make a matrix positive definite
function make_posdef(A::SparseMatrixCSC)
    # Symmetrize and add diagonal dominance
    S = (A + A') / 2
    d = vec(sum(abs, S; dims = 2))
    return S + spdiagm(0 => d .+ 1.0)
end

# Handle Symmetric wrapper from MatrixDepot
make_posdef(A::Symmetric) = make_posdef(sparse(A))

# Test matrices from SSMC (larger for meaningful benchmarks)
test_matrices = [
    ("HB/bcsstk15", "Structural, n=3948"),
    ("HB/bcsstk16", "Structural, n=4884"),
    ("HB/bcsstk17", "Structural, n=10974"),
    ("HB/bcsstk18", "Structural, n=11948"),
]

println("\nTest matrices:")
for (name, desc) in test_matrices
    println("  - $name ($desc)")
end

results = []

for (matrix_name, desc) in test_matrices
    println("\n" * "="^80)
    println("Matrix: $matrix_name ($desc)")
    println("="^80)

    # Load and prepare matrix
    try
        A_raw = matrixdepot(matrix_name)
        Q = make_posdef(A_raw)
        n = size(Q, 1)

        println("  Size: $n × $n")
        println("  Nonzeros: $(nnz(Q))")

        # Create mean vector and evaluation point
        Random.seed!(42)
        μ = randn(n)
        z = randn(n)

        # Create GMRF (baseline)
        println("\n  Creating GMRF...")
        gmrf = GMRF(μ, Q, LinearSolve.CHOLMODFactorization())

        # Create ChordalGMRF
        println("  Creating ChordalGMRF...")
        chordal_gmrf = ChordalGMRF(μ, Q)

        # Correctness check
        println("\n  Correctness check:")
        lpdf_gmrf = logpdf(gmrf, z)
        lpdf_chordal = logpdf(chordal_gmrf, z)
        abs_diff = abs(lpdf_gmrf - lpdf_chordal)
        rel_diff = abs_diff / (abs(lpdf_gmrf) + 1.0e-10)

        println("    GMRF logpdf:        $(@sprintf("%.8f", lpdf_gmrf))")
        println("    ChordalGMRF logpdf: $(@sprintf("%.8f", lpdf_chordal))")
        println("    Absolute diff:      $(@sprintf("%.2e", abs_diff))")
        println("    Relative diff:      $(@sprintf("%.2e", rel_diff))")

        correct = rel_diff < 1.0e-8
        println("    Match: $(correct ? "✓ YES" : "✗ NO")")

        # Performance benchmark
        println("\n  Performance benchmark:")

        # Benchmark GMRF
        print("    GMRF...        ")
        bench_gmrf = @benchmark logpdf($gmrf, $z) samples = 20 seconds = 5
        time_gmrf = minimum(bench_gmrf.times) / 1.0e6
        println("$(@sprintf("%.3f", time_gmrf)) ms")

        # Benchmark ChordalGMRF
        print("    ChordalGMRF... ")
        bench_chordal = @benchmark logpdf($chordal_gmrf, $z) samples = 20 seconds = 5
        time_chordal = minimum(bench_chordal.times) / 1.0e6
        println("$(@sprintf("%.3f", time_chordal)) ms")

        speedup = time_gmrf / time_chordal
        println("    Speedup: $(@sprintf("%.2f", speedup))×")

        # Gradient correctness check
        println("\n  Gradient correctness check (w.r.t. z):")
        gmrf_logpdf = x -> logpdf(gmrf, x)
        chordal_logpdf = x -> logpdf(chordal_gmrf, x)

        # Use prepared gradients for both backends
        prep_gmrf = DifferentiationInterface.prepare_gradient(gmrf_logpdf, AutoZygote(), z)
        prep_chordal = DifferentiationInterface.prepare_gradient(chordal_logpdf, AutoMooncake(; config = nothing), z)

        grad_gmrf = DifferentiationInterface.gradient(gmrf_logpdf, prep_gmrf, AutoZygote(), z)
        grad_chordal = DifferentiationInterface.gradient(chordal_logpdf, prep_chordal, AutoMooncake(; config = nothing), z)
        grad_abs_diff = norm(grad_gmrf - grad_chordal)
        grad_rel_diff = grad_abs_diff / (norm(grad_gmrf) + 1.0e-10)

        println("    Absolute diff: $(@sprintf("%.2e", grad_abs_diff))")
        println("    Relative diff: $(@sprintf("%.2e", grad_rel_diff))")

        grad_correct = grad_rel_diff < 1.0e-8
        println("    Match: $(grad_correct ? "✓ YES" : "✗ NO")")

        # Gradient performance benchmark
        println("\n  Gradient performance benchmark (via DifferentiationInterface, prepared):")

        print("    GMRF...        ")
        bench_grad_gmrf = @benchmark DifferentiationInterface.gradient($gmrf_logpdf, $prep_gmrf, AutoZygote(), $z) samples = 20 seconds = 5
        time_grad_gmrf = minimum(bench_grad_gmrf.times) / 1.0e6
        println("$(@sprintf("%.3f", time_grad_gmrf)) ms")

        print("    ChordalGMRF... ")
        bench_grad_chordal = @benchmark DifferentiationInterface.gradient($chordal_logpdf, $prep_chordal, AutoMooncake(; config = nothing), $z) samples = 20 seconds = 5
        time_grad_chordal = minimum(bench_grad_chordal.times) / 1.0e6
        println("$(@sprintf("%.3f", time_grad_chordal)) ms")

        grad_speedup = time_grad_gmrf / time_grad_chordal
        println("    Speedup: $(@sprintf("%.2f", grad_speedup))×")

        push!(
            results, (
                name = matrix_name,
                n = n,
                nnz = nnz(Q),
                correct = correct,
                grad_correct = grad_correct,
                time_gmrf = time_gmrf,
                time_chordal = time_chordal,
                speedup = speedup,
                time_grad_gmrf = time_grad_gmrf,
                time_grad_chordal = time_grad_chordal,
                grad_speedup = grad_speedup,
            )
        )

    catch e
        println("  ✗ Failed: $(typeof(e).name.name): $(sprint(showerror, e; context = :limit => true))")
        push!(
            results, (
                name = matrix_name, n = 0, nnz = 0, correct = false, grad_correct = false,
                time_gmrf = NaN, time_chordal = NaN, speedup = NaN,
                time_grad_gmrf = NaN, time_grad_chordal = NaN, grad_speedup = NaN,
            )
        )
    end
end

# Summary table
println("\n" * "="^80)
println("SUMMARY: FORWARD PASS")
println("="^80)

println("\n" * "-"^95)
@printf(
    "%-20s %8s %10s %8s %12s %12s %10s\n",
    "Matrix", "n", "nnz", "Correct", "GMRF (ms)", "Chordal (ms)", "Speedup"
)
println("-"^95)

for r in results
    correct_str = r.correct ? "✓" : "✗"
    @printf(
        "%-20s %8d %10d %8s %12.3f %12.3f %10.2f×\n",
        r.name, r.n, r.nnz, correct_str, r.time_gmrf, r.time_chordal, r.speedup
    )
end
println("-"^95)

# Gradient summary table
println("\n" * "="^80)
println("SUMMARY: GRADIENT (via DifferentiationInterface, prepared)")
println("="^80)

println("\n" * "-"^95)
@printf(
    "%-20s %8s %10s %8s %12s %12s %10s\n",
    "Matrix", "n", "nnz", "Correct", "GMRF (ms)", "Chordal (ms)", "Speedup"
)
println("-"^95)

for r in results
    correct_str = r.grad_correct ? "✓" : "✗"
    @printf(
        "%-20s %8d %10d %8s %12.3f %12.3f %10.2f×\n",
        r.name, r.n, r.nnz, correct_str, r.time_grad_gmrf, r.time_grad_chordal, r.grad_speedup
    )
end
println("-"^95)

# Overall stats
valid_results = filter(r -> !isnan(r.speedup), results)
if !isempty(valid_results)
    avg_speedup = sum(r.speedup for r in valid_results) / length(valid_results)
    avg_grad_speedup = sum(r.grad_speedup for r in valid_results) / length(valid_results)
    all_correct = all(r.correct for r in valid_results)
    all_grad_correct = all(r.grad_correct for r in valid_results)

    println("\nOverall:")
    println("  Forward - All match: $(all_correct ? "✓ YES" : "✗ NO"), Avg speedup: $(@sprintf("%.2f", avg_speedup))×")
    println("  Gradient - All match: $(all_grad_correct ? "✓ YES" : "✗ NO"), Avg speedup: $(@sprintf("%.2f", avg_grad_speedup))×")
end

println("\n" * "="^80)
println("BENCHMARK COMPLETE")
println("="^80)
