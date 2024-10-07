export AbstractStoppingCriterion, NewtonDecrementCriterion, StepNumberCriterion, OrCriterion

abstract type AbstractStoppingCriterion end

struct NewtonDecrementCriterion <: AbstractStoppingCriterion
    threshold::Real

    function NewtonDecrementCriterion(threshold::Real = 1e-6)
        new(threshold)
    end
end

struct StepNumberCriterion <: AbstractStoppingCriterion
    max_steps::Int

    function StepNumberCriterion(max_steps::Int = 20)
        new(max_steps)
    end
end

struct OrCriterion <: AbstractStoppingCriterion
    criteria::Vector{<:AbstractStoppingCriterion}

    function OrCriterion(criteria::Vector{<:AbstractStoppingCriterion})
        new(criteria)
    end
end
