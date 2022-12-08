## running Euler integrator for the mathematical pendulum
######################################
######################################
######################################

#Juno.clearconsole()    #maybe activate again

using Pkg
Pkg.add("PrettyTables")
Pkg.add("Plots")
Pkg.add("Printf")
Pkg.add("DelimitedFiles")
using PrettyTables
using Plots, Printf
using DelimitedFiles

## booleans
show_video = false

## starting the file
println("-- pendulum euler --")

## load pendulum
include("Dynsys.jl")
pendulum = Dynsys.Math_pendulum(1.0, 10.0, 1.0, 500, 1, 0.0)
include("Integrator.jl") # added by Aaron
println(typeof(pendulum))
## load integrator and memory for the results

Integ = Dynsys.Integrator(1.0e-3,1000)
Integ.res_phi = zeros(Integ.timesteps)
Integ.res_phi_dot = zeros(Integ.timesteps)
time = zeros(Integ.timesteps)
#Integ::Main.Dynsys.Integrator #added by Aaron

## run time integration
# initial setting
fig = Dynsys.create_fig(pendulum)
Dynsys.plot_state(pendulum)
display(fig)
# running over the time step
for i in 1:Integ.timesteps
    # integration step
    global pendulum = run_step(Integ, "central_diff", pendulum) # paste in "euler" or "central_diff"
    time[i] = Integ.delta_t * i
    # (homework)
    println(typeof(pendulum))
    # plot the state
    fig = Dynsys.create_fig(pendulum) # Why does this not work? How to call pendulum?
    Dynsys.run_step(Integ, "euler", pendulum)
    Dynsys.plot_state(pendulum)
    display(fig)
    # save the step
    Integ.res_phi[i] = pendulum.phi
    Integ.res_phi_dot[i] = pendulum.phi_dot
end
println(time)
display(plot(time,Integ.res_phi))
#show(plot(time,Integ.res_phi))
readline()
display(plot(time,Integ.res_phi_dot))
readline()
######## Homework
# implement the euler integration step
# implement the central difference integration step
# plot coordinate phi and the time derivative phi_dot
