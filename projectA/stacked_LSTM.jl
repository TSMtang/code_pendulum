using PrettyTables
using Plots, Printf
using DelimitedFiles
using Flux, Statistics, ProgressMeter
using MLDataUtils


## load data
data = readdlm("data.txt", '\t', Float64, '\n')
t = data[:,1]
phi = data[:,2]
phi_dot = data[:,3]
delta_t = t[1]


## initialize
x = Float32.(t)
y = Float32.(phi)
# LSTM setup
in_size = 50
out_size = 5
# dataset split setup
split_train = 0.5
split_valid = 0.25


## prepare dataset
# normal split of dataset
at = split_train
(x_train, y_train), (x_vali_test, y_vali_test) = splitobs((x, y); at = at)
at = split_valid / (1-split_train)
(x_validation, y_validation), (x_test, y_test) = splitobs((x_vali_test, y_vali_test); at = at)

# treat data as only step sequences
training = Array(y_train)
validation = Array(y_validation)
test = Array(y_test)

# dataset for LSTM
function LSTMdataset_prepare(dataset, in_size, out_size)
    # Prepare LSTM x_train and y_train
    # dataset:      sequence of training data [one dimension vector]
    # in_size:      input neural size [int]
    # out_size:     output neural size [int]
    # The idea is to use in_size datapoints to predict the following out_size datapoints
    # inputs:       inputs of LSTM NN [matrix in_size*samples]
    # outputs:      outputs of LSTM NN [matrix in_size*samples]
    sample = size(dataset)[1] - (in_size + out_size) + 1
    inputs = Matrix{Float32}(undef,in_size,0)
    outputs = Matrix{Float32}(undef,out_size,0)
    for i in 1:sample
        input = dataset[i : i + in_size - 1]
        output = dataset[i + in_size : i + in_size + out_size - 1]
        inputs = [inputs input]
        outputs = [outputs output]
    end
    return inputs, outputs
end

x_train, y_train = LSTMdataset_prepare(training, in_size, out_size)

# reshape dataset size(dataset) = (sequence data points, features, samples)
x_train = reshape(x_train, (size(x_train)[1], 1, size(x_train)[2]))
y_train = reshape(y_train, (size(y_train)[1], 1, size(y_train)[2]))


## Neural network
# model
model = Chain(
    LSTM(in_size => 20),
    LSTM(20 => 20),
    Dense(20 =>out_size),
    )

# huperparameters
pars = Flux.params(model)  # contains references to arrays in model
opt = Flux.Adam(0.01)      # will store optimiser momentum, etc.
loader = Flux.DataLoader((x_train, y_train), batchsize=10, shuffle=true)


## Training loop
loss_train = []
loss_valid = []
@showprogress for epoch in 1:250
    loss_train_epoch = 0
    loss_valid_epoch = 0
    for (x, y) in loader
        loss, grad = Flux.withgradient(pars) do
            # Evaluate model and loss inside gradient context:
            y_hat = model(x)
            Flux.mse(y_hat, y)
        end
        Flux.update!(opt, pars, grad)
        loss_train_epoch += loss
    end
    # validation loss
    #y_hat_validation = model(x_validation)
    #loss_valid_epoch = Flux.mse(y_hat_validation, y_validation)
    # logging
    push!(loss_train, loss_train_epoch)
    #push!(loss_valid, loss_valid_epoch)
end

## predict and test
function LSTM_predict(train_data, model, in_size, out_size, predict_size)
    # predict future data with LSTM model
    # train_data: training data [one dim sequence]
    # model: LSTM model
    # in_size: input neuron size (int)
    # out_size: output neuron size (int)
    # predict_size: how many steps to predict (int)
    # predict (return): predicted data [one dim vector]
    input = train_data[end-in_size+1 : end]
    predict = []
    for i in 1:out_size:predict_size
        # predict and save
        output = model(input)               # predict
        predict = vcat(predict, output)     # save predicted value
        # move to the next step
        input = input[out_size + 1 : end]   # remove pased steps
        input = vcat(input, output)         # add future steps
    end
    return predict
end

# train predict
x_train_predict = (in_size + out_size : size(training)[1]) * delta_t
y_train_predict = reshape(model(x_train)[out_size,:,:], size(x_train)[3])
# test predict
predict_size = 1000
x_test_predict = (size(x_train)[3] + in_size + 1 : size(x_train)[3] + in_size + predict_size) * delta_t
y_test_predict = LSTM_predict(training, model, in_size, out_size, predict_size)


## plot results
fig = plot(x, y, xlabel = "time (s)", ylabel = "phi", label = "true")
plot!(x_train_predict, y_train_predict, label = "train_predict")
plot!(x_test_predict, y_test_predict, label = "test_predict")
display(fig)
savefig("stacked_LSTM.png")