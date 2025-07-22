"""
    autodiff.jl

Main include file for automatic differentiation functionality in GaussianMarkovRandomFields.jl.

This module provides efficient ChainRulesCore-based automatic differentiation rules for:
- GMRF construction
- logpdf computation
- And other core GMRF operations

The implementation uses selected inverses via SelectedInversion.jl to compute gradients
efficiently without materializing full covariance matrices.
"""

# Import ChainRulesCore for rrule definitions
using ChainRulesCore

# Include specific autodiff rule implementations
include("constructors.jl")
include("logpdf.jl")