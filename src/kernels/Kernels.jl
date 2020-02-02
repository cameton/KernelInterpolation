
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
abstract type Kernel <: Function end

"""
    evaluate(k::Kernel, x::AbstractVector, y::AbstractVector)

Evaluate the kernel `k` on `x` and `y`. 
Requires `length(x) == length(y)`.
"""
evaluate(k::Kernel, x::AbstractVector, y::AbstractVector) = k(x,y)
evaluate(k::Kernel, x::AbstractFloat, y::AbstractFloat) = k(x,y)

resulttype(k::Kernel, T, S) =
    promote_type(T, S, paramtypes(k)...)

function _colwise!()
end

function colwise!()
end

function colwise()
end

function _pairwise!(
    Out::AbstractMatrix,
    k::Kernel,
    A::AbstractMatrix,
    B::AbstractMatrix,
)
    (size(Out, 1) == size(A, 2) && size(Out, 2) == size(B, 2)) ||
        throw(DimensionMismatch(
            "output size $(size(Out)) should be ($(size(A, 2)), $(size(B, 2))"
        ))
    dim1, dim2 = axes(Out)
    @inbounds for c = dim2
        b = view(B, :, c)
        for r = dim1
            Out[r, c] = k(view(A, :, r), b)
        end
    end
    Out
end

function _pairwise!(
    Out::AbstractMatrix,
    k::Kernel,
    A::AbstractMatrix,
)
    size(Out, 1) == size(Out, 2) == size(A, 2) ||
        throw(DimensionMismatch(
            "output size $(size(Out)) should be ($(size(A, 2)), $(size(A, 2))"
        ))
    dim2 = axes(Out, 2)
    @inbounds for c = dim2
        a = view(B, :, c)
        
        for r = 1:(c - 1)
            Out[r, c] = Out[c, r]
        end
        
        Out[c, c] = h(a, a)
        
        for r = (c + 1):dim2.stop
            Out[r, c] = k(view(A, :, r), a)
        end
    end
    Out
end

function pairwise!(
    Out::AbstractMatrix,
    k::Kernel,
    A::AbstractMatrix,
    B::AbstractMatrix;
    dims::Integer,
)
    dims == 1 || dims == 2 ||
        throw(ArgumentError("dim $dims is not 1 or 2"))
    d = dims == 1 ? 2 : 1
    size(A, d) == size(B, d) ||
        throw(ArgumentError("dim $d of A does not match dim $d of B"))
    dims == 2 ? _pairwise!(Out, k, A, B) : _pairwise!(Out, k, A', B')
end

function pairwise!(
    Out::AbstractMatrix,
    k::Kernel,
    A::AbstractMatrix;
    dims::Integer,
)
    dims == 1 || dims == 2 ||
        throw(ArgumentError("dim $dims is not 1 or 2"))
    dims == 2 ? _pairwise!(Out, k, A) : _pairwise!(Out, k, A')
end

function pairwise(
    k::Kernel,
    A::AbstractMatrix,
    B::AbstractMatrix;
    dims::Integer
)
    dims == 1 || dims == 2 ||
        throw(ArgumentError("dim $dims is not 1 or 2"))
    T = resulttype(k, eltype(A), eltype(B))
    Out = Matrix{T}(undef, size(A, dims), size(B, dims))
    pairwise!(Out, k, A, B; dims=dims)
    return Out
end

function pairwise(
    k::Kernel,
    A::AbstractMatrix;
    dims::Integer
)
    dims == 1 || dims == 2 ||
        throw(ArgumentError("dim $dims is not 1 or 2"))
    T = resulttype(k, eltype(A), eltype(A))
    Out = Matrix{T}(undef, size(A, dims), size(A, dims))
    pairwise!(Out, k, A; dims=dims)
    return Out
end

"""
    TODO
"""
abstract type RadialKernel{D<:SemiMetric} <: Kernel end

@inline (k::RadialKernel)(x::AbstractVector, y::AbstractVector) = k(k.dist(x, y))
@inline (k::RadialKernel)(x::AbstractFloat, y::AbstractFloat) = k(k.dist(x, y))

"""
    TODO
"""
abstract type ZonalKernel{M<:AbstractPDMat} <: Kernel end

(k::ZonalKernel)(x::AbstractVector, y::AbstractVector) =
    k(dot(x, k.M * y))
(k::ZonalKernel)(x::AbstractFloat, y::AbstractFloat) =
    k(dot(x, k.M * y))

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
    colwise,
    colwise!,
    pairwise,
    pairwise!,
    numparams,
    paramtypes,
    params,
    setparams!,
    
    RadialKernel,
    Exponential,
    SqExponential,
    Gaussian,
    SqExponentialARD,
    GaussianARD,
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
    Spherical,
    
    ZonalKernel

end # module
