
"""
    SquaredExponential(γ::Uniform) # TODO fix this doc

The squared exponential kernel (aka the Gaussian kernel or the radial basis
function kernel) with a length scale 
parameter `γ` given by a continuous uniform distribution.

```math
k_\\theta(\\tau) = \\exp\\Big\\lbrace -\\Big(\\frac{\\tau}{\\theta}\\Big)^2\\Big\\rbrace
```

```julia    
SquaredExponential(Uniform()) # γ is uniform over [0, 1]
SquaredExponential(Uniform(a, b)) # γ is uniform over [a, b]

sampleparam(k) # A random sample of the parameters of k
k(θ, τ) # The kernel evaluated on the distance τ with parameters θ
```

External links
* [RBF Kernel on Wikipedia](https://en.wikipedia.org/wiki/Radial_basis_function_kernel)

"""
struct SquaredExponential{R} <: RadialBasisFunction{R} end

SquaredExponential() = SquaredExponential{Float64}()

function (k::SquaredExponential{R})((γ,)::Tuple{R}, τ::R) where R <: Real
    return exp(-(τ / γ)^2)
end

function paramnames(k::SquaredExponential)
    return (:γ,)
end

function paramdomain(k::SquaredExponential)
    return (γ = (0, Inf),)
end
