
using PrettyTables
using Plots, Printf
using DelimitedFiles
using Flux, Statistics, ProgressMeter
#using FluxUtils
using FFTW
using MLDataUtils
#=
## Example 1, 1D ode
@parameters t phi
@variables u(..)
@derivatives Dt'~t

# 1D ODE
eq = Dtt(phi(t)) ~ phi(t)
=#
NNODE = Chain(x -> [x], # Take in a scalar and transform it into an array
           Dense(1 => 32,tanh),
           Dense(32 => 1),
           first) # Take first value, i.e. return a scalar
NNODE(1.0)

g(t) = t*NNODE(t) + 1f0
using Statistics
ϵ = sqrt(eps(Float32))
loss() = mean(abs2(((g(t+ϵ)-g(t))/ϵ) - cos(2π*t)) for t in 0:1f-2:1f0)
opt = Flux.Descent(0.01)
data = Iterators.repeated((), 5000)
iter = 0
cb = function () #callback function to observe training
  global iter += 1
  if iter % 500 == 0
    display(loss())
  end
end
display(loss())
Flux.train!(loss, Flux.params(NNODE), data, opt; cb=cb)
using Plots
t = 0:0.001:1.0
plot(t,g.(t),label="NN")
plot!(t,1.0 .+ sin.(2π.*t)/2π, label = "True Solution")
savefig("example1.png")