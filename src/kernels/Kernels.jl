module Kernels

using LinearAlgebra
using Distances
using PDMats

abstract type Kernel end
abstract type RadialKernel{D<:SemiMetric} <: Kernel end
abstract type ZonalKernel{M<:AbstractPDMat} <: Kernel end

const kernel = [
    "KernelSum",
    "KernelProduct",
    # "AllSubsets",
    # "ANOVA",
    # "Graph",
    # "Diffusion",
]

const radial = [
    "SquaredExponential",
    "Exponential",
    "Matern",
    "Spherical",
    "PolyharmonicSpline",
    "Multiquadric",
    "InverseMultiquadric",
    "RationalQuadratic",
    "Periodic",
    "LocallyPeriodic",
    "Laplace",
]

const zonal = [
    "Polynomial",
    "Legendre",
    "Poisson",
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

end # module
