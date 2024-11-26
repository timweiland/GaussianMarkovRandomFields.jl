export AbstractStoppingCriterion, NewtonDecrementCriterion, StepNumberCriterion, OrCriterion

"""
    AbstractStoppingCriterion

Abstract type for the specification of a criterion that tells an optimization
algorithm when to stop iterating.
"""
abstract type AbstractStoppingCriterion end

"""
    NewtonDecrementCriterion(threshold)

Stops the optimization procedure when
∇f(xₖ) ∇²f(xₖ)⁻¹ ∇f(xₖ) < threshold.
"""
struct NewtonDecrementCriterion <: AbstractStoppingCriterion
    threshold::Real

    function NewtonDecrementCriterion(threshold::Real = 1e-6)
        new(threshold)
    end
end

"""
    StepNumberCriterion(max_steps)

Stops the optimization procedure when a maximum number of iterations / steps is
reached.
"""
struct StepNumberCriterion <: AbstractStoppingCriterion
    max_steps::Int

    function StepNumberCriterion(max_steps::Int = 20)
        new(max_steps)
    end
end

"""
    OrCriterion(criteria)

Stops the optimization procedure when any of the criteria in `criteria` are
fulfilled.
"""
struct OrCriterion <: AbstractStoppingCriterion
    criteria::Vector{<:AbstractStoppingCriterion}

    function OrCriterion(criteria::Vector{<:AbstractStoppingCriterion})
        new(criteria)
    end
end
