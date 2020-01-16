
"""
    SqExponentialARD(lnℒ::AbstractVector) 
    <: RadialKernel{WeightedSqEuclidean, 1}

The squared exponential kernel (aka the Gaussian kernel) with
vector valued length scale
``\\exp(2 \\ln\\mathscr{L}_i) = \\mathscr{L}_i^2 > 0``.

Finding an optimal ``\\mathscr{L}`` is refered to as automatic 
relevance detetermination. 

For the matrix ``\\Sigma_{ii} = \\mathscr{L}_i^{-2}``

```math
k_{\\mathscr{L}}(x, y) = k_{\\mathscr{L}}(\\lVert x - y\\rVert_\\Sigma^2) = 
\\exp\\Big\\lbrace -\\frac{1}{2}\\lVert x - y\\rVert_\\Sigma^2\\Big\\rbrace
```

"""
mutable struct SqExponentialARD{
    F<:AbstractFloat,
    V<:AbstractVector{F},
} <: RadialKernel{WeightedSqEuclidean, 1}
    dist::WeightedSqEuclidean
    lnℒ::V
    diagΣ::V
end

function SqExponentialARD(lnℒ::AbstractVector)
    diagΣ = exp.(-2 .* lnℒ)
    SqExponentialARD(WeightedSqEuclidean(diagΣ), lnℒ, diagΣ)
end

(k::SqExponentialARD)(τ::AbstractFloat) = exp(-τ / 2)

GaussianARD = SqExponentialARD

params(k::SqExponentialARD) = (lnℒ = k.lnℒ,)
function setparams!(k::SqExponentialARD, lnℒ::AbstractVector)
    diagΣ = exp.(-2 .* lnℒ)
    k.dist = WeightedSqEuclidean(diagΣ)
    k.lnℒ = lnℒ
    k.diagΣ = diagΣ
end
