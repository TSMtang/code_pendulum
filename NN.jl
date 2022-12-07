using Plots
using Statistics
using Flux
using Pkg       #Package for reading/opening xlsx-files
using MLUtils   #Package defines interfaces and implements common utilities for Machine Learning pipelines.
using DataFrames
using CSV

#Reading the pendulum data
#import XLSX

#time_input  = XLSX.readdata("pendulum_data.xlsx", "time", "A2:B4")
#position_output = XLSX.readdata("pendulum_data.xlsx", "position", "A2:B4")

pendulum_data = CSV.read("/Users/franzoldopp/Projekte/CIE/code_pendulum_franz/code_pendulum/pendulum.csv", DataFrame)

X = pendulum_data.time
Y = pendulum_data.pos

#function generate_fake_data(n)
#    θ  = 2*π*rand(1,n)
#    r  = rand(1,n)/3
#    x1 = @. r*cos(θ)
#    x2 = @. r*sin(θ)+0.5
#    return vcat(x1,x2)
#end

# Creating our data --> 35/65 split
data_pendulum_train, data_pendulum_test = splitobs(pendulum_data, at = 0.35)
print(time_input_train)

# Visualizing
scatter(time_input, position_output)

function NeuralNetwork()
    return Chain(                               #Chain(): Iterate through any number of iterators in sequence
            Dense(2, 25,relu),                  #Dense(): Dense(in::Integer, out::Integer, σ = identity)
            Dense(25,1,x->σ.(x))
            )
end

# Organizing the data in batches --> using single batch
X    = hcat(real, fake)                              #hcat: this concatenates along dimension 2.
Y    = vcat(ones(train_size), zeros(train_size))     #vcat: this concatenates along dimension 1.
data = Flux.Data.DataLoader((X, Y), batchsize=100, shuffle=true);

# Defining our model, optimization algorithm and loss function
m    = NeuralNetwork()
opt = Descent(0.05) #we should use ADAM here
loss(x, y) = sum(Flux.Losses.binarycrossentropy(m(x), y))

# Training Method 1
ps = Flux.params(m)
epochs = 20
for i in 1:epochs
    Flux.train!(loss, ps, data, opt)
end
println("Mean real: ", + mean(m(real)), ' ', "Mean fake: ", +  mean(m(fake))) # Print model prediction

# Visualizing the model predictions
scatter(real[1,1:100], real[2,1:100], zcolor=m(real)')
scatter!(fake[1,1:100], fake[2,1:100], zcolor=m(fake)', legend=false)