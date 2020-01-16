
"""
    TODO
"""
module Kernels

using LinearAlgebra
using Distances
using PDMats
using SpecialFunctions
using StatsFuns

"""
    Kernel{N}

Abstract supertype for kernel functions with `N` real vector or scalar
valued parameters.
"""
abstract type Kernel{N} <: Function end

"""
    evaluate(k::Kernel, x::AbstractVector, y::AbstractVector)

Evaluate the kernel `k` on `x` and `y`. 
Requires `length(x) == length(y)`.
"""
evaluate(k::Kernel, x::AbstractVector, y::AbstractVector) = k(x,y)

"""
    TODO
"""
abstract type RadialKernel{D<:SemiMetric, N} <: Kernel{N} end

(k::RadialKernel)(x::AbstractVector, y::AbstractVector) = k(k.dist(x, y))

"""
    TODO
"""
abstract type ZonalKernel{M<:AbstractPDMat, N} <: Kernel{N} end

(k::ZonalKernel)(x::AbstractVector, y::AbstractVector) = k(dot(x, k.M * y))

const kernel = [
    "KernelSum",
    "KernelProduct",
    # "AllSubsets",
    # "ANOVA",
    # "Graph",
    # "Diffusion",
]

const radial = [
    "Exponential",
    "SqExponential",
    "SqExponentialARD",
    "GammaExponential",
    "RationalQuadratic",
    "Matern",
    "HalfInteger",
    "Multiquadric",
    "InverseMultiquadric",
    "ThinPlateSpline",
    "Cubic",
    "PolyharmonicSpline",
    "Periodic",
    "Laplace",
    "Spherical",
]

const zonal = [
    "Linear",
    "Polynomial",
    "Legendre",
    "Poisson",
    "RestrictedRadial",
]

for f in kernel
    include("$(f).jl")
end

for f in radial
    include(joinpath("radial", "$(f).jl"))
end

for f in zonal
    include(joinpath("zonal", "$(f).jl"))
end

export
    Kernel,
    evaluate,
    RadialKernel,
    ZonalKernel,
    Exponential,
    SqExponential,
    SqExponentialARD,
    GammaExponential,
    RationalQuadratic,
    Matern,
    HalfInteger,
    Multiquadric,
    InverseMultiquadric,
    ThinPlateSpline,
    Cubic,
    PolyharmonicSpline,
    Periodic,
    Laplace,
    Spherical

end # module
