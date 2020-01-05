module Kernel

using LinearAlgebra
using Distances

abstract type KernelFunction{R<:Real} end

evaluate(k::KernelFunction{R}, θ::Tuple{Vararg{R}}, x, y) where R <: Real = k(θ, x, y)

# TODO colwise & pairwise

abstract type RadialBasisFunction{R} <: KernelFunction{R} end

evaluate(k::RadialBasisFunction{R}, θ::Tuple{Vararg{R}}, τ) where R <: Real = k(θ, τ)

function (k::RadialBasisFunction{R})(
    θ::Tuple{Vararg{R}},
    x::AbstractVector{R},
    y::AbstractVector{R}
) where R <: Real
    return k(θ, norm(x - y))
end

(k::RadialBasisFunction{R})(θ::Tuple{Vararg{R}}, x::R, y::R) where R <: Real =  k(θ, norm(x - y))

# TODO colwise functions

function pairwise!(
    Out::AbstractMatrix{R},
    k::RadialBasisFunction{R},
    θ::Tuple{Vararg{R}},
    X::AbstractMatrix{R},
    Y::AbstractMatrix{R};
    dims::Int
) where R <: Real
    Distances.pairwise!(Out, Euclidean(), X, Y, dims=dims)
    Out .= k.(Ref(θ), Out)
    Out
end

function pairwise!(
    Out::AbstractMatrix{R},
    k::RadialBasisFunction{R},
    θ::Tuple{Vararg{R}},
    X::AbstractMatrix{R};
    dims::Int
) where R <: Real
    Distances.pairwise!(Out, Euclidean(), X, dims=dims)
    Out .= k.(Ref(θ), Out)
    Out
end

function pairwise(
    k::RadialBasisFunction{R},
    θ::Tuple{Vararg{R}},
    X::AbstractMatrix{R},
    Y::AbstractMatrix{R};
    dims::Int
) where R <: Real
    Out = Matrix{R}(undef, size(X, dims), size(Y, dims))
    pairwise!(Out, k, θ, X, Y, dims=dims)
    return Out
end

function pairwise(
    k::RadialBasisFunction{R},
    θ::Tuple{Vararg{R}},
    X::AbstractMatrix{R};
    dims::Int
) where R <: Real
    Out = Matrix{R}(undef, size(X, dims), size(X, dims))
    pairwise!(Out, k, θ, X, dims=dims)
    return Out
end


const kernels = [
]

const rbfkernels = [
    "SquaredExponential"
]

for f in kernels
    include(joinpath("kernel", "$(f).jl"))
end

for f in rbfkernels
    include(joinpath("kernel", "radialbasis", "$(f).jl"))
end

end # module
