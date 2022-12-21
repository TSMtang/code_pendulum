## Train neural network and predict the data
#####################################
#####################################
#####################################

using PrettyTables
using Plots, Printf
using DelimitedFiles
using Flux, Statistics, ProgressMeter
using MLDataUtils
using AutoGrad

## load data
data = readdlm("data.txt", '\t', Float64, '\n')
t = data[:,1]
phi = data[:,2]
phi_dot = data[:,3]

## initialize
x = Float32.(t)
y = Float32.(phi)


## prepare dataset
(x_train, y_train), (x_vali_test, y_vali_test) = splitobs((x, y); at = 0.35)
(x_valid, y_valid), (x_test, y_test) = splitobs((x_vali_test, y_vali_test); at = 0.54)

# reshape dataset
x_train = reshape(x_train, (1, length(x_train)))
x_valid = reshape(x_valid, (1, length(x_valid)))
x_test = reshape(x_test, (1, length(x_test)))

y_train = reshape(y_train, (1, length(y_train)))
y_valid = reshape(y_valid, (1, length(y_valid)))
y_test = reshape(y_test, (1, length(y_test)))


# model
model = Chain(
    Dense(1 => 10, tanh),
    Dense(10 => 10, tanh),
    Dense(10 => 1),
    )

# huperparameters
pars = Flux.params(model)  # contains references to arrays in model
opt = Flux.Adam(0.01)      # will store optimiser momentum, etc.
loader = Flux.DataLoader((x_train, y_train), batchsize=100, shuffle=true)

# phisics loss
function PhysicsLoss(phi)
    # physics parameters
    l, g, m, c, delta_t = (10/300, 10, 1, 5, 0.001)

    # calculate MSE
    loss = []
    stepsize = 1
    for i in 1+stepsize : 50 : length(phi)-stepsize
        phi_dot = (phi[i+stepsize] - phi[i-stepsize]) / (2 * delta_t * stepsize)
        phi_dotdot = (phi[i+stepsize] - 2*phi[i] + phi[i-stepsize]) / (delta_t * stepsize)^2
        r = phi_dotdot + c * phi_dot + g/l * phi[i]
        loss = vcat(loss, [r])
    end
    return mean(loss.^2) / 50000
end

# Training loop
loss_train = []
loss_valid = []
@showprogress for epoch in 1:10000
    loss_train_epoch = 0
    loss_valid_epoch = 0
    for (x, y) in loader
        loss, grad = Flux.withgradient(pars) do
            # Evaluate model and loss inside gradient context:
            y_hat = model(x)
            r_NN = Flux.mse(y_hat, y)

            # Evaluate physics loss
            phi = model(hcat(x_train, x_valid)) 
            r_physics = PhysicsLoss(phi)

            r_NN + r_physics
        end
        Flux.update!(opt, pars, grad)
        loss_train_epoch += loss
    end
    # validation loss
    y_hat_valid = model(x_valid)
    loss_valid_epoch = Flux.mse(y_hat_valid, y_valid)
    # logging
    push!(loss_train, loss_train_epoch)
    push!(loss_valid, loss_valid_epoch)
end

# predict and test
x = hcat(x_train, x_valid, x_test)
y_true = phi
y_hat_train = model(x_train)
y_hat_valid = model(x_valid)
y_hat_test = model(x_test)

## plot results
# reshape for plot
function ReshapeForPlot(v)
    return reshape(v, (length(v)))
end

# plot loss
fig = plot(loss_train, xlabel = "epoch", ylabel = "loss", label = "train")
plot!(loss_valid, label = "valid")
display(fig)

# plot prediction
fig = plot(ReshapeForPlot(x), ReshapeForPlot(y_true), xlabel = "time (s)", ylabel = "phi", label = "true")
plot!(ReshapeForPlot(x_train), ReshapeForPlot(y_hat_train), xlabel = "time (s)", ylabel = "phi", label = "predict_train")
plot!(ReshapeForPlot(x_valid), ReshapeForPlot(y_hat_valid), xlabel = "time (s)", ylabel = "phi", label = "predict_valid")
plot!(ReshapeForPlot(x_test), ReshapeForPlot(y_hat_test), xlabel = "time (s)", ylabel = "phi", label = "predict_test")
display(fig)