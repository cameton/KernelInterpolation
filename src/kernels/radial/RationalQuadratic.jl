
"""
    RationalQuadratic(lnℓ::AbstractFloat) <: RadialKernel{SqEuclidean, 1}

The rational quadratic kernel with parameters 
``\\exp(2 \\ln\\ell) = \\ell^2 > 0`` and ``\\exp(\\ln\\alpha) = \\alpha > 0``. 

The rational quadratic kernel may be used to model functions with a 
variable length scale.

```math
k_{\\ell,\\alpha}(x, y) = k_{\\ell,\\alpha}(\\lVert x - y\\rVert_2^2) = 
\\bigg(1 + \\frac{\\lVert x - y\\rVert_2^2}{2\\alpha\\ell^2}\\bigg)^{-\\alpha}
```

External links
* [Rational quadratic covariance function on Wikipedia](https://en.wikipedia.org/wiki/Rational_quadratic_covariance_function)

"""
mutable struct RationalQuadratic{
    F<:AbstractFloat
} <: RadialKernel{SqEuclidean, 2}
    dist::SqEuclidean
    lnℓ::F
    lnα::F
    
    α::F
    twoαℓ²::F
    function RationalQuadratic(
        dist::SqEuclidean,
        lnℓ::AbstractFloat,
        lnα::AbstractFloat
    )
        lnℓ, lnα = promote(lnℓ, lnα)
        new{typeof(lnℓ)}(
            dist,
            lnℓ,
            lnα,
            exp(lnα),
            exp(2 * lnℓ + lnα + float(logtwo)),
        )
    end
end

RationalQuadratic(lnℓ::AbstractFloat, lnα::AbstractFloat) =
    RationalQuadratic(SqEuclidean(), lnℓ, lnα)

(k::RationalQuadratic)(τ::AbstractFloat) = (1 + τ / k.twoαℓ²)^(-k.α)

params(k::RationalQuadratic) = (lnℓ = k.lnℓ, lnα = k.lnα)
function setparams!(k::RationalQuadratic, lnℓ::AbstractFloat, lnα::AbstractFloat)
    k.lnℓ = lnℓ
    k.lnα = lnα
    k.α = exp(lnα)
    k.twoαℓ² = exp(2 * lnℓ + lnα + float(logtwo))
end
