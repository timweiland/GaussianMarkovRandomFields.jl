# Internals

```@meta
CurrentModule = GaussianMarkovRandomFields
```

These functions and types are implementation details of GaussianMarkovRandomFields.jl.
They are not exported and may change without notice; they are documented here for
contributors and for anyone extending the package.

## Discretizations and state-space models

```@docs
assemble_mass_matrix
assemble_diffusion_matrix
assemble_advection_matrix
lump_matrix
assemble_streamline_diffusion_matrix
apply_soft_constraints!
JointSSMMatrices
joint_ssm
ImplicitEulerSSM
ImplicitEulerJointSSMMatrices
```

## KL-optimal sparse approximate Cholesky

```@docs
sparse_approximate_cholesky
sparse_approximate_cholesky!
reverse_maximin_ordering
reverse_maximin_ordering_and_sparsity_pattern
sparsity_pattern_from_ordering
SupernodeClustering
form_supernodes
PermutedMatrix
```

## Exponential families

```@docs
ExponentialFamilyLikelihood
NormalLikelihood
PoissonLikelihood
BernoulliLikelihood
BinomialLikelihood
NegBinLikelihood
GammaLikelihood
StudentTLikelihood
apply_link
apply_invlink
```

## Workspace internals

```@docs
ensure_numeric!
ensure_selinv!
dimension
ConstraintInfo
WorkspaceBackend
```
