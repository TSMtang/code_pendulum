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
    #res_phi_dot_dot::Vector # added by Aaron
end

## run one integration time step
function run_step(int::Main.Dynsys.Integrator, type, pendulum)
    if type == "euler"
        run_euler_step(int, pendulum)
    elseif type == "central_diff"
        run_central_diff_step(int, pendulum)             # Here stood mp instead of int like it stand on the "euler" part
    else
        println("... integration type not understood ...")
    end
end

## euler integration time step (homework)
function run_euler_step(int::Main.Dynsys.Integrator, pendulum)
    println("Running euler step")
    ###### (homework) ######
    # without damping constant c at the moment
    phi_dot_dot = - pendulum.g / pendulum.l * pendulum.phi
    println(phi_dot_dot)
    phi_dot_plus = pendulum.phi_dot + int.delta_t * phi_dot_dot
    println(phi_dot_plus)
    phi_plus = pendulum.phi + int.delta_t * pendulum.phi_dot
    pendulum.phi = phi_plus
    pendulum.phi_dot = phi_dot_plus
    
    return pendulum
    
end

## central difference time step (homework)
function run_central_diff_step(int::Main.Dynsys.Integrator, pendulum)
    println("Running central difference step")
    ###### (homework) ######
    phi_dot_dot = - pendulum.g / pendulum.l * pendulum.phi - pendulum.phi_dot * pendulum.c
    phi_i_plus = pendulum.phi + pendulum.phi_dot * int.delta_t + phi_dot_dot * int.delta_t^2 / 2
    phi_i_minus = pendulum.phi - pendulum.phi_dot * int.delta_t + phi_dot_dot * int.delta_t^2 / 2
    phi_dot_plus = (phi_i_plus - phi_i_minus)/(2 * int.delta_t)
    pendulum.phi = phi_i_plus
    pendulum.phi_dot = phi_dot_plus
    return pendulum
end
