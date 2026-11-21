# R-INLA side of the benchmark. For each mesh level: rebuild the shared mesh, read the
# counts simulated by fit_gmrfs.jl, and time a Gaussian approximation of the posterior with
# all latent marginals at the same fixed hyperparameters.
#
#   Rscript fit_rinla.R <workdir>   (run after fit_gmrfs.jl has written y.csv)
#
# Single-threaded: with fixed hyperparameters, R-INLA's outer threads have nothing to
# parallelise, and its inner threads only apply to the PARDISO solver.

suppressPackageStartupMessages(library(INLA))

argv <- commandArgs(trailingOnly = TRUE)
out <- if (length(argv) >= 1) argv[1] else "_workdir"

RANGE <- 0.3
SIGMA <- 1.0

# The inla binary's own timing of the approximate inference: the computation itself,
# without R-side preparation, process start-up, or reading and writing files.
inference_seconds <- function(res) {
    line <- grep("Approx inference \\(total\\)", res$cpu.intern, value = TRUE)
    as.numeric(sub(".*:\\s*([0-9.]+) seconds.*", "\\1", line))
}

levels <- sort(as.integer(sub("level_", "", list.files(out, pattern = "^level_"))))
rows <- list()
for (level in levels) {
    d <- file.path(out, sprintf("level_%d", level))
    nodes <- read.csv(file.path(d, "nodes.csv"))
    tris <- read.csv(file.path(d, "triangles.csv"))
    mesh <- inla.mesh.create(loc = as.matrix(nodes[, c("x", "y")]), tv = as.matrix(tris[, c("v1", "v2", "v3")]))
    stopifnot(mesh$n == nrow(nodes), max(abs(mesh$loc[, 1:2] - as.matrix(nodes[, c("x", "y")]))) == 0)
    obs <- read.csv(file.path(d, "obs_coords.csv"))
    y <- read.csv(file.path(d, "y.csv"))$y

    # alpha = 2 matches smoothness = 0 (nu = 1) in 2D. Priors are irrelevant: the
    # hyperparameters are fixed below.
    spde <- inla.spde2.pcmatern(mesh, alpha = 2, prior.range = c(RANGE, 0.5), prior.sigma = c(SIGMA, 0.5))
    A <- inla.spde.make.A(mesh, loc = as.matrix(obs[, c("s1", "s2")]))
    stk <- inla.stack(data = list(y = y), A = list(A), effects = list(spatial = seq_len(mesh$n)), tag = "est")

    fit <- function() {
        inla(
            y ~ -1 + f(spatial, model = spde),
            data = inla.stack.data(stk), family = "poisson",
            control.predictor = list(A = inla.stack.A(stk), compute = FALSE),
            # Fixed hyperparameters (internal scale: log range, log sigma), and a Gaussian
            # approximation for the latent marginals without the variational-Bayes mean
            # correction: the same computation as fit_gmrfs.jl.
            control.mode = list(theta = c(log(RANGE), log(SIGMA)), fixed = TRUE),
            control.inla = list(strategy = "gaussian", int.strategy = "eb", control.vb = list(enable = FALSE)),
            num.threads = "1:1"
        )
    }

    n_rep <- if (mesh$n > 20000) 3 else 7
    walls <- numeric(n_rep)
    running <- numeric(n_rep)
    compute <- numeric(n_rep)
    for (r in seq_len(n_rep)) {
        walls[r] <- system.time(res <- fit())[["elapsed"]]
        running[r] <- as.numeric(res$cpu.used[["Running"]])
        compute[r] <- inference_seconds(res)
    }
    fs <- res$summary.random$spatial
    write.csv(data.frame(mean = fs$mean, sd = fs$sd), file.path(d, "rinla_field_summary.csv"), row.names = FALSE)

    rows[[length(rows) + 1]] <- data.frame(
        level = level, n = mesh$n, m = length(y),
        t_median = median(walls), t_min = min(walls), running_median = median(running),
        compute_median = median(compute)
    )
    print(rows[[length(rows)]])
}
write.csv(do.call(rbind, rows), file.path(out, "rinla_times.csv"), row.names = FALSE)
cat(sprintf("INLA version %s\n", as.character(packageVersion("INLA"))))
