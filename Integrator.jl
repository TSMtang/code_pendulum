####################################
# Explicit Euler
#
# numeric integration file for the
# mathematical pendulum
#
# - explicit euler
# -
####################################


mutable struct Integrator
    delta_t::Float64
    timesteps::Int64
    Integrator(delta_t, timesteps) = new(delta_t, timesteps)
    res_phi::Vector
    res_phi_dot::Vector
end

## run one integration time step
function run_step(int::Integrator, type, pendulum, step)
    if type == "euler"
        run_euler_step(int, pendulum, step)
    elseif type == "central_diff"
        run_central_diff_step(int, pendulum, step)
    else
        println("... integration type not understood ...")
    end
end

## euler integration time step (homework)
function run_euler_step(int::Integrator, pendulum, step)
    println("Running euler step $step")
    ###### (phi -> phi_2) ######
    # input
    g = pendulum.g
    l = pendulum.l
    c = pendulum.c
    delta_t = int.delta_t
    phi = pendulum.phi
    phi_dot = pendulum.phi_dot
    # calculation
    phi_dotdot = - g / l * phi - c * phi_dot
    phi_dot_2 = phi_dot + delta_t * phi_dotdot
    phi_2 = phi + delta_t * phi_dot
    # output
    pendulum.phi = phi_2
    pendulum.phi_dot = phi_dot_2
end

## central difference time step (homework)
function run_central_diff_step(int::Integrator, pendulum, step)
    println("Running central difference step $step")
    ###### (phi_0 -> phi_1) ######
    # input
    g = pendulum.g
    l = pendulum.l
    c = pendulum.c
    delta_t = int.delta_t
    phi_0 = pendulum.phi
    phi_dot_0 = pendulum.phi_dot
    ## calculation
    # initial condition of phi_minus1 (previous point)
    phi_dotdot_0 = - g / l * phi_0 - c * phi_dot_0
    phi_minus1 = phi_0 - delta_t * phi_dot_0 + 1/2 * delta_t^2 * phi_dotdot_0
    # central difference method
    phi_1 = ( (phi_0 * (- delta_t^2 * g / l + 2) + phi_minus1 * (delta_t * c / 2 - 1))
             /(delta_t * c / 2 + 1) )
    phi_2 = ( (phi_1 * (- delta_t^2 * g / l + 2) + phi_0 * (delta_t * c / 2 - 1))
             /(delta_t * c / 2 + 1) )
    phi_dot_1 = (phi_2 - phi_0) / (2 * delta_t)
    # output
    pendulum.phi = phi_1
    pendulum.phi_dot = phi_dot_1
end
