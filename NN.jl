using DataFrames
using CSV
using Flux
using CUDA
using Plots
using Statistics
using Pkg       #Package for reading/opening xlsx-files
using MLUtils   #Package defines interfaces and implements common utilities for Machine Learning pipelines.


pendulum_output = CSV.read("C:\\Users\\Proteus\\Desktop\\RWTH\\WISE23\\Code_Pendulum\\CIE\\ProjectA\\code_pendulum\\pendulum.csv", DataFrame)

X = pendulum_output.time
Y = pendulum_output.pos

# Visualizing
# scatter(time_input, position_output)

model = Chain(
    Dense(1,5),   # activation function inside layer
    BatchNorm(3, relu),
    Dense(5,1),
    BatchNorm(3,relu),
    ) |> gpu 

# # Organizing the data in batches --> using single batch
# X    = hcat(real, fake)                              #hcat: this concatenates along dimension 2.
# Y    = vcat(ones(train_size), zeros(train_size))     #vcat: this concatenates along dimension 1.
# data = Flux.Data.DataLoader((X, Y), batchsize=100, shuffle=true);

# # Defining our model, optimization algorithm and loss function
# m    = NeuralNetwork()
# opt = Descent(0.05) #we should use ADAM here
# loss(x, y) = sum(Flux.Losses.binarycrossentropy(m(x), y))

# # Training Method 1
# ps = Flux.params(m)
# epochs = 20
# for i in 1:epochs
#     Flux.train!(loss, ps, data, opt)
# end
# println("Mean real: ", + mean(m(real)), ' ', "Mean fake: ", +  mean(m(fake))) # Print model prediction

# # Visualizing the model predictions
# scatter(real[1,1:100], real[2,1:100], zcolor=m(real)')
# scatter!(fake[1,1:100], fake[2,1:100], zcolor=m(fake)', legend=false)
