
"""
    Matern(lnℓ::AbstractFloat, ln𝓋::AbstractFloat) <: RadialKernel{SqEuclidean}

The matern kernel with parameters 
``\\exp(\\ln\\ell) = \\ell > 0`` and 
``\\exp(\\ln\\mathscr{v}) = \\mathscr{v} > 0``. 



```math
k_{\\ell,\\mathscr{v}}(x, y) 
= k_{\\ell,\\mathscr{v}}(\\lVert x - y\\rVert_2) = 
\\frac{2^{1 - \\mathscr{v}}}{\\Gamma(\\mathscr{v})}
\\bigg(\\sqrt{2\\mathscr{v}}\\frac{\\lVert x - y\\rVert_2}{\\ell}
\\bigg)^\\mathscr{v}K_\\mathscr{v}
\\bigg(\\sqrt{2\\mathscr{v}}\\frac{\\lVert x - y\\rVert_2}{\\ell}\\bigg)
```

External links
* [Matérn covariance function on Wikipedia](https://en.wikipedia.org/wiki/Matérn_covariance_function)

"""
mutable struct Matern{
    F<:AbstractFloat
} <: RadialKernel{Euclidean}
    dist::Euclidean
    lnℓ::F
    ln𝓋::F
    
    𝓋::F
    c::F # 2^(1 - 𝓋) / Γ(𝓋)
    sqrttwo𝓋_ℓ::F
    
    function Matern(
        dist::Euclidean,
        lnℓ::AbstractFloat,
        ln𝓋::AbstractFloat
    )
        lnℓ, ln𝓋 = promote(lnℓ, ln𝓋)
        𝓋 = exp(ln𝓋)
        new{typeof(lnℓ)}(
            dist,
            lnℓ,
            ln𝓋,
            𝓋,
            exp2(1 - 𝓋) / gamma(𝓋),
            exp((float(logtwo) + ln𝓋) / 2),
        )
    end
end

Matern(lnℓ::AbstractFloat, ln𝓋::AbstractFloat) =
    Matern(Euclidean(), lnℓ, ln𝓋)

@inline function (k::Matern)(τ::AbstractFloat)
    if τ == 0
        return oneunit(promote(τ, k.𝓋)[1])
    end
    t = k.sqrttwo𝓋_ℓ * τ
    return k.c * (t^k.𝓋) * besselk(k.𝓋, t)
end

numparams(::Matern) = (1, 1)
paramtypes(::Matern{F}) where F = (F, F)
params(k::Matern) = (lnℓ = k.lnℓ, ln𝓋 = k.ln𝓋)
function setparams!(k::Matern{F}, lnℓ::F, ln𝓋::F) where F
    𝓋 = exp(ln𝓋)
    k.lnℓ = lnℓ
    k.ln𝓋 = ln𝓋
    k.𝓋 = 𝓋
    k.c = exp2(1 - 𝓋) / gamma(𝓋)
    k.sqrttwo𝓋_ℓ = exp((float(logtwo) + ln𝓋) / 2 - lnℓ)
end
