# Compare the two engines' latent posteriors node by node, and collect the timings.
#
#   julia --project=paper/benchmark paper/benchmark/compare.jl <workdir>

using CSV, DataFrames

workdir = ARGS[1]
gt = CSV.read(joinpath(workdir, "gmrfs_times.csv"), DataFrame)
rt = CSV.read(joinpath(workdir, "rinla_times.csv"), DataFrame)
rows = DataFrame()
for (g, r) in zip(eachrow(gt), eachrow(rt))
    d = joinpath(workdir, "level_$(g.level)")
    gs = CSV.read(joinpath(d, "gmrfs_field_summary.csv"), DataFrame)
    rs = CSV.read(joinpath(d, "rinla_field_summary.csv"), DataFrame)
    push!(
        rows, (;
            g.n, g.m, gmrfs_s = g.t_median, rinla_s = r.t_median, rinla_running_s = r.running_median, rinla_compute_s = r.compute_median,
            speedup = r.t_median / g.t_median, speedup_compute = r.compute_median / g.t_median, max_abs_dmean = maximum(abs.(gs.mean .- rs.mean)),
            max_rel_dsd = maximum(abs.(gs.sd .- rs.sd) ./ rs.sd),
        )
    )
end
show(rows; allcols = true)
println()
CSV.write(joinpath(@__DIR__, "results.csv"), rows)
