module KernelInterpolation

using Reexport

include("./Kernel.jl")

@reexport using .Kernel

end # module
