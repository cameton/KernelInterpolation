
"""
    SqExponential(lnℓ::AbstractFloat) <: RadialKernel{SqEuclidean}

The squared exponential kernel (aka the Gaussian kernel) with length scale 
``\\exp(2 \\ln\\ell) = \\ell^2 > 0``. 

```math
k_{\\ell}(x, y) = k_{\\ell}(\\lVert x - y\\rVert_2^2) = 
\\exp\\bigg\\lbrace -\\frac{\\lVert x - y\\rVert_2^2}{2\\ell^2}\\bigg\\rbrace
```

External links
* [RBF Kernel on Wikipedia](https://en.wikipedia.org/wiki/Radial_basis_function_kernel)

"""
mutable struct SqExponential{F<:AbstractFloat} <: RadialKernel{SqEuclidean}
    dist::SqEuclidean
    lnℓ::F
    twoℓ²::F
    function SqExponential(dist::SqEuclidean, lnℓ::AbstractFloat)
        new{typeof(lnℓ)}(dist, lnℓ, exp(2 * lnℓ + float(logtwo)))
    end
end

SqExponential(lnℓ::AbstractFloat) = SqExponential(SqEuclidean(), lnℓ)

@inline (k::SqExponential)(τ::AbstractFloat) = exp(-τ / k.twoℓ²)

Gaussian = SqExponential

numparams(::SqExponential) = (1,)
paramtypes(::SqExponential{F}) where F = (F,)
params(k::SqExponential) = (lnℓ = k.lnℓ,)
function setparams!(k::SqExponential{F}, lnℓ::F) where F
    k.lnℓ = lnℓ
    k.twoℓ² = exp(2 * lnℓ + float(logtwo))
end
