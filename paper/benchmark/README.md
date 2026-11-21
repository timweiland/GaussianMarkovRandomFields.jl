# Benchmark: Gaussian approximation against R-INLA

Times one Gaussian approximation of the posterior, including all latent marginal
variances, at fixed hyperparameters, in GaussianMarkovRandomFields.jl and in R-INLA.
The model is a Poisson likelihood with a Matérn SPDE prior (α = 2, range 0.3, marginal
standard deviation 1) on a sequence of shared meshes. Both engines solve on the same
mesh, fit the same counts, and compute the same quantity: R-INLA runs with fixed
hyperparameters, the `"gaussian"` strategy and the variational-Bayes mean correction
disabled. `compare.jl` checks that the two latent posteriors agree node by node.

Both engines run single-threaded. With fixed hyperparameters, R-INLA's outer threads
have nothing to parallelise, and its inner threads only apply to the PARDISO solver.

```sh
Rscript paper/benchmark/gen_meshes.R paper/benchmark/_workdir
julia --project=paper/benchmark paper/benchmark/fit_gmrfs.jl paper/benchmark/_workdir
(cd paper/benchmark && Rscript fit_rinla.R _workdir)
julia --project=paper/benchmark paper/benchmark/compare.jl paper/benchmark/_workdir
```

`fit_gmrfs.jl` also simulates the counts and writes them for `fit_rinla.R`, so it must
run first. Times are medians over repeated fits, warm for GaussianMarkovRandomFields.jl
(after compilation; the first fit is reported separately as `t_first`). For R-INLA,
`results.csv` reports three times:

- `rinla_s`: the whole `inla()` call, as a user experiences it.
- `rinla_running_s`: the external `inla` binary, without R-side preparation and
  post-processing, but including process start-up and file input and output.
- `rinla_compute_s`: the binary's own timing of its approximate inference
  ("Approx inference (total)" in `cpu.intern`), which is the computation itself and the
  fair counterpart of the GaussianMarkovRandomFields.jl time.

Without a PARDISO licence, R-INLA uses its built-in TAUCS solver, as it did here.

`results.csv` was produced on an Apple M1 Max with Julia 1.10.10 and R-INLA 25.6.7.
