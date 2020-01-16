
"""
    Exponential(lnℓ::AbstractFloat) <: RadialKernel{Euclidean, 1}

The exponential kernel with length scale 
``\\exp(\\ln \\ell) = \\ell > 0``.

```math
k_{\\ell}(x, y) = k_{\\ell}(\\lVert x - y\\rVert_2) = 
\\exp\\bigg\\lbrace -\\frac{\\lVert x - y\\rVert_2}{\\ell}\\bigg\\rbrace
```
"""
mutable struct Exponential{F<:AbstractFloat} <: RadialKernel{Euclidean, 1}
    dist::Euclidean
    lnℓ::F
    ℓ::F
    function Exponential(dist::Euclidean, lnℓ::AbstractFloat)
        new{typeof(lnℓ)}(dist, lnℓ, exp(lnℓ))
    end
end

Exponential(lnℓ::AbstractFloat) = Exponential(Euclidean(), lnℓ)

(k::Exponential)(τ::AbstractFloat) = exp(-τ / k.ℓ)

params(k::Exponential) = (lnℓ = k.lnℓ,)
function setparams!(k::Exponential, lnℓ::AbstractFloat)
    k.lnℓ = lnℓ
    k.ℓ = exp(lnℓ)
end


