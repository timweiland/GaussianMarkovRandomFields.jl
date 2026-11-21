# Generate a refinable sequence of SPDE meshes shared by both engines. Each level has an
# observation grid and an inla.mesh.2d at decreasing max.edge, so the latent dimension n
# grows. Nodes, triangles and observation coordinates are written per level; both
# GaussianMarkovRandomFields.jl and R-INLA then solve on exactly this mesh.
#
#   Rscript gen_meshes.R <workdir>

suppressPackageStartupMessages(library(INLA))

argv <- commandArgs(trailingOnly = TRUE)
out <- if (length(argv) >= 1) argv[1] else "_workdir"
dir.create(out, showWarnings = FALSE, recursive = TRUE)

# Per level: side length of the observation grid, and the inner mesh max.edge.
gms <- c(20, 35, 55, 90, 140, 220)
edges <- c(0.100, 0.055, 0.032, 0.018, 0.011, 0.007)

for (i in seq_along(gms)) {
    g <- seq(0.03, 0.97, length.out = gms[i])
    coords <- as.matrix(expand.grid(s1 = g, s2 = g))
    me <- edges[i]
    mesh <- inla.mesh.2d(loc = coords, max.edge = c(me, me * 3), cutoff = me * 0.5, offset = c(me * 2, 0.2))
    d <- file.path(out, sprintf("level_%d", i))
    dir.create(d, showWarnings = FALSE, recursive = TRUE)
    write.csv(data.frame(x = mesh$loc[, 1], y = mesh$loc[, 2]), file.path(d, "nodes.csv"), row.names = FALSE)
    tv <- mesh$graph$tv
    write.csv(data.frame(v1 = tv[, 1], v2 = tv[, 2], v3 = tv[, 3]), file.path(d, "triangles.csv"), row.names = FALSE)
    write.csv(data.frame(s1 = coords[, 1], s2 = coords[, 2]), file.path(d, "obs_coords.csv"), row.names = FALSE)
    cat(sprintf("level %d: %d observations, %d mesh nodes\n", i, nrow(coords), mesh$n))
}
