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
using Flux
using CSV
using DataFrames


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
x = LinRange(0, Integ.timesteps*Integ.delta_t, Integ.timesteps)
# x = x * Integ.delta_t
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



## Training a neural network
# training dataset
data_pen = DataFrame(time = x,pos = y1)
CSV.write("C:\\Users\\Proteus\\Desktop\\RWTH\\WISE23\\Code_Pendulum\\CIE\\ProjectA\\code_pendulum\\pendulum.csv",data_pen)