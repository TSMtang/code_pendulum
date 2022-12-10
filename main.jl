## running Euler integrator for the mathematical pendulum
######################################
######################################
######################################

#Juno.clearconsole()
# import Pkg
# Pkg.add("packagename")

using PrettyTables
using Plots, Printf
using DelimitedFiles

## booleans
show_video = true
anim_pendulum = Animation()

## initialization
int_method = "central_diff"
delta_t = 1.0e-3
steps = 2000

## starting the file
println("-- pendulum ", int_method, " --")

## load pendulum
include("Dynsys.jl")
pendulum = Dynsys.Math_pendulum(10/300, 10.0, 1.0, 5, 1.0, 0.0)

## load integrator and memory for the results
Integ = Dynsys.Integrator(delta_t,steps)
Integ.res_phi = zeros(Integ.timesteps)
Integ.res_phi_dot = zeros(Integ.timesteps)

## run time integration
# initial setting
fig = Dynsys.create_fig(pendulum)
Dynsys.plot_state(pendulum)
display(fig)
# running over the time step
for i in 1:Integ.timesteps
    # integration step
    step = i
    # plot the state
    fig = Dynsys.create_fig(pendulum)
    Dynsys.run_step(Integ, int_method, pendulum, step)
    Dynsys.plot_state(pendulum)
    display(fig)
    frame(anim_pendulum)
    # save the step
    Integ.res_phi[i] = pendulum.phi
    Integ.res_phi_dot[i] = pendulum.phi_dot
end

## plot coordinate phi and the time derivative phi_dot
# time
x = LinRange(1, Integ.timesteps, Integ.timesteps)
x = x * Integ.delta_t
# phi
y1 = Integ.res_phi
display(plot(x, y1, xlabel = "time (s)", ylabel = "phi", label = "phi"))
savefig("phi.png")
# phi_dot
y2 = Integ.res_phi_dot
display(plot(x, y2, label = "phi_dot"))
xlabel!("time(s)")
ylabel!("phi_dot (1/s)")
savefig("phi_dot.png")

if show_video
    gif(anim_pendulum, "pendulum.gif")
end

## write data
open("data.txt", "w") do io
    writedlm(io, [x y1 y2])
end

## Training a neural network
using Flux, Statistics, ProgressMeter
# prepare dataset
training_points = round(Int, 0.35 * steps)

x_train = range(1,training_points) * delta_t            |> collect
x_test = range(training_points + 1, steps) * delta_t    |> collect
y_train = Integ.res_phi[1 : training_points]
y_test = Integ.res_phi[training_points + 1 : end]

x_train = reshape(x_train, (1, length(x_train)))
x_test = reshape(x_test, (1, length(x_test)))
y_train = reshape(y_train, (1, length(y_train)))
y_test = reshape(y_test, (1, length(y_test)))

# model
model = Chain(
    Dense(1 => 5, tanh),   # activation function inside layer
    BatchNorm(5),
    Dense(5 => 5, tanh),
    BatchNorm(5),
    Dense(5 => 1),
    )

# huperparameters
pars = Flux.params(model)  # contains references to arrays in model
opt = Flux.Adam(0.01)      # will store optimiser momentum, etc.
loader = Flux.DataLoader((x_train, y_train), batchsize=100, shuffle=true)

# Training loop
losses = []
@showprogress for epoch in 1:1_000
    for (x, y) in loader
        loss, grad = Flux.withgradient(pars) do
            # Evaluate model and loss inside gradient context:
            y_hat = model(x)
            Flux.mse(y_hat, y)
        end
        Flux.update!(opt, pars, grad)
        push!(losses, loss)  # logging, outside gradient context
    end
end

# predict and test
x = hcat(x_train, x_test)
y_predict = model(x)
y_true = hcat(y_train, y_test)

## plot results
x = reshape(x, (length(x)))
y_predict = reshape(y_predict, (length(y_predict)))
y_true = reshape(y_true, (length(y_true)))
plot(x, y_predict, xlabel = "time (s)", ylabel = "phi", label = "predict")
plot!(x, y_true, xlabel = "time (s)", ylabel = "phi", label = "true")