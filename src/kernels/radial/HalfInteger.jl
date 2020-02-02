
"""
    HalfInteger(lnℓ::AbstractFloat, p::Integer) <: RadialKernel{SqEuclidean}

The matern kernel with
``\\exp(\\ln\\ell) = \\ell > 0`` and 
``p + \\frac{1}{2} = \\mathscr{v} 
\\in \\Big\\lbrace \\frac{1}{2}, \\frac{3}{2}, \\ldots\\Big\\rbrace``. 



```math
k_{\\ell,p}(x, y) 
= k_{\\ell,p}(\\lVert x - y\\rVert_2) = 
\\exp\\bigg(−\\sqrt{2\\mathscr{v}}\\frac{\\lVert x - y\\rVert_2}{\\ell}\\bigg)
\\sum^p_{i = 0}\\binom{p}{i}\\frac{(p + i)!}{(2p)!}
\\bigg(\\sqrt{8\\mathscr{v}}\\frac{\\lVert x - y\\rVert_2}{\\ell}\\bigg)^{p - i}
```

External links
* [Matérn covariance function on Wikipedia](https://en.wikipedia.org/wiki/Matérn_covariance_function)

"""
mutable struct HalfInteger{
    F<:AbstractFloat,
    Z<:Integer
} <: RadialKernel{Euclidean}
    dist::Euclidean
    lnℓ::F
    p::Z
    
    sqrttwo𝓋_ℓ::F
    
    function HalfInteger(
        dist::Euclidean,
        lnℓ::AbstractFloat,
        p::Integer
    )
        new{typeof(lnℓ), typeof(p)}(
            dist,
            lnℓ,
            p,
            sqrt(2 * p + oneunit(lnℓ)) * exp(-lnℓ),
        )
    end
end

HalfInteger(lnℓ::AbstractFloat, p::Integer) =
    HalfInteger(Euclidean(), lnℓ, p)

@inline function (k::HalfInteger)(τ::AbstractFloat)
    acc = 1
    coef = 1
    for j=k.p:-1:1
        coef *= (2 * k.sqrttwo𝓋_ℓ * τ) / (k.p + j)
        acc += binomial(k.p, j) * coef
    end
    return exp(-k.sqrttwo𝓋_ℓ * τ) * acc
end

numparams(::HalfInteger) = (1, 1)
paramtypes(::HalfInteger{F, Z}) where F where Z = (F, Z)
params(k::HalfInteger) = (lnℓ = k.lnℓ, p = k.p)
function setparams!(k::HalfInteger{F,Z}, lnℓ::F, p::Z) where F where Z
    k.lnℓ = lnℓ
    k.p = p
    k.sqrttwo𝓋_ℓ = sqrt(2 * p + oneunit(lnℓ)) * exp(-lnℓ)
end
