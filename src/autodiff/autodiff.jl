"""
    autodiff.jl

Main include file for automatic differentiation functionality in GaussianMarkovRandomFields.jl.

This module provides efficient ChainRulesCore-based automatic differentiation rules for:
- GMRF construction
- logpdf computation
- gaussian_approximation (Fisher scoring optimization)
- And other core GMRF operations

The implementation uses selected inverses via SelectedInversion.jl to compute gradients
efficiently without materializing full covariance matrices.
"""

# Import ChainRulesCore for rrule definitions
using ChainRulesCore

# Include specific autodiff rule implementations (reverse-mode)
include("constructors.jl")
include("precision_gradient.jl")  # Helper for computing precision gradients
include("logpdf.jl")
include("gaussian_approximation.jl")
