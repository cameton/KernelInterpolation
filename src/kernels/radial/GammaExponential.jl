
"""
    GammaExponential(lnℓ::AbstractFloat, lnγ::AbstractFloat) 
    <: RadialKernel{Euclidean, 2}

The gamma exponential kernel with length scale 
``\\exp(\\gamma\\ln\\ell) = \\ell^\\gamma > 0``.

```math
k_{\\gamma,\\ell}(x, y) = 
k_{\\gamma,\\ell}(\\lVert x - y\\rVert_2) = 
\\exp\\bigg\\lbrace -
\\frac{\\lVert x - y\\rVert_2^\\gamma}{\\ell^\\gamma}\\bigg\\rbrace
```
"""
mutable struct GammaExponential{F<:AbstractFloat} <: RadialKernel{Euclidean, 2}
    dist::Euclidean
    lnℓ::F
    lnγ::F
    
    ℓᵞ::F
    γ::F
    function GammaExponential(
        dist::Euclidean,
        lnℓ::AbstractFloat,
        lnγ::AbstractFloat
    )
        lnℓ, lnγ = promote(lnℓ, lnγ)
        γ = exp(lnγ)
        new{typeof(lnℓ)}(dist, lnℓ, lnγ, exp(γ * lnℓ), γ)
    end
end

GammaExponential(lnℓ::AbstractFloat, lnγ::AbstractFloat) =
    GammaExponential(Euclidean(), lnℓ, lnγ)

(k::GammaExponential)(τ::AbstractFloat) = exp(-τ^k.γ / k.ℓᵞ)

params(k::GammaExponential) = (lnℓ = k.lnℓ, lnγ = k.lnγ)
function setparams!(k::GammaExponential, lnℓ::AbstractFloat, lnγ::AbstractFloat)
    lnℓ, lnγ = promote(lnℓ, lnγ)
    γ = exp(lnγ)
    k.lnℓ = lnℓ
    k.lnγ = lnγ
    k.ℓᵞ = exp(γ * lnℓ)
    k.γ = γ
end
