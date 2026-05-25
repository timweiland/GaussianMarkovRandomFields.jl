#!/usr/bin/env julia
"""
Run the regression benchmark suite and print a summary.

Usage:
    cd benchmarks
    julia --project=. run_regression.jl              # Just print results
    julia --project=. run_regression.jl results.json # Also save raw results to JSON

The JSON output can be compared against a baseline using
`BenchmarkTools.judge(estimated, baseline)`.
"""

using BenchmarkTools

include("benchmarks.jl")

println("="^80)
println("REGRESSION BENCHMARKS — GaussianMarkovRandomFields.jl")
println("="^80)
println()
println("Warming up...")
warmup(SUITE)

println("Running benchmarks...")
results = run(SUITE; verbose = true)

println()
println("="^80)
println("RESULTS")
println("="^80)

for (group_name, group) in sort(collect(pairs(results)); by = first)
    println()
    println("[", group_name, "]")
    for (bench_name, trial) in sort(collect(pairs(group)); by = first)
        med = median(trial)
        mem = trial.memory
        allocs = trial.allocs
        println(
            "  ", rpad(bench_name, 32), " ",
            lpad(BenchmarkTools.prettytime(BenchmarkTools.time(med)), 12),
            "   ", lpad(string(allocs), 8), " allocs",
            "   ", lpad(BenchmarkTools.prettymemory(mem), 10)
        )
    end
end

if length(ARGS) >= 1
    out_path = ARGS[1]
    println()
    println("Saving raw results to ", out_path)
    BenchmarkTools.save(out_path, results)
end

println()
println("="^80)
println("DONE")
println("="^80)
