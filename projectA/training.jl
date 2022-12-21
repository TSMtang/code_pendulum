## Train neural network and predict the data
#####################################
#####################################
#####################################
#import Pkg; Pkg.add("ProgressMeter")
#import Pkg; Pkg.add("MLDataUtils")
#import Pkg; Pkg.add("FFTW")
using PrettyTables
using Plots, Printf
using DelimitedFiles
using Flux, Statistics, ProgressMeter
#using FluxUtils
using FFTW
using MLDataUtils

## load data
data = readdlm("data.txt", '\t', Float64, '\n')
t = data[:,1]
phi = data[:,2]
phi_dot = data[:,3]

## initialize
x = Float32.(t)
y = Float32.(phi)


## prepare dataset
(x_train, y_train), (x_vali_test, y_vali_test) = splitobs((x, y); at = 0.5)
(x_validation, y_validation), (x_test, y_test) = splitobs((x_vali_test, y_vali_test); at = 0.5)

# reshape dataset
x_train = reshape(x_train, (1, length(x_train)))
x_validation = reshape(x_validation, (1, length(x_validation)))
x_test = reshape(x_test, (1, length(x_test)))

y_train = reshape(y_train, (1, length(y_train)))
y_validation = reshape(y_validation, (1, length(y_validation)))
y_test = reshape(y_test, (1, length(y_test)))


# model
model = Chain(
    Dense(1 => 5, tanh),
    Dense(5 => 5, tanh),
    Dense(5 => 1),
    )

# huperparameters
pars = Flux.params(model)  # contains references to arrays in model
opt = Flux.Adam(0.01)      # will store optimiser momentum, etc.
loader = Flux.DataLoader((x_train, y_train), batchsize=100, shuffle=true)
#loader = Flux.DataLoader((x_train, y_train), batchsize=1, shuffle=false)

# Training loop
#loss(x, y) = Flux.mse(model(x), y)
#Flux.train!(loss, pars, (x_train, y_train), opt)

# Training loop
loss_train = []
loss_valid = []
@showprogress for epoch in 1:1000
    loss_train_epoch = 0
    loss_valid_epoch = 0
    for (x, y) in loader
        loss, grad = Flux.withgradient(pars) do
            # Evaluate model and loss inside gradient context:
            y_hat = model(x)
            Flux.mse(y_hat, y)           #modifying the loss function ak
        end
        Flux.update!(opt, pars, grad)
        loss_train_epoch += loss
    end
    # validation loss
    y_hat_validation = model(x_validation)
    loss_valid_epoch = Flux.mse(y_hat_validation, y_validation)
    # logging
    push!(loss_train, loss_train_epoch)
    push!(loss_valid, loss_valid_epoch)
end

###
#loader = Flux.DataLoader((transpose(x), transpose(y)), batchsize=1, shuffle=true)
#predict = []
#for (x, y) in loader
#    y_hat = model(x)
#    push!(predict, y_hat[1,1])
#end
###

#y_hat_validation = model(x_validation)
#loss_validation = Flux.mse(y_hat_validation, y_validation)

# predict and test
x = hcat(x_train, x_validation, x_test)
y_predict = model(x)
y_true = phi

## plot results
# reshape
x = reshape(x, (length(x)))
y_predict = reshape(y_predict, (length(y_predict)))
y_true = reshape(y_true, (length(y_true)))
# plot loss
fig = plot(loss_train, xlabel = "epoch", ylabel = "loss", label = "train")
plot!(loss_valid, label = "valid")
display(fig)
savefig("training_loss.png")
# plot prediction
fig = plot(x, y_predict, xlabel = "time (s)", ylabel = "phi", label = "predict")
plot!(x, y_true, xlabel = "time (s)", ylabel = "phi", label = "true")
display(fig)
savefig("training_prediction.png")