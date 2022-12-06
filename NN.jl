using DataFrames
using CSV
using Flux
using CUDA

pendulum_output = CSV.read("C:\\Users\\Proteus\\Desktop\\RWTH\\WISE23\\Code_Pendulum\\CIE\\ProjectA\\code_pendulum\\pendulum.csv", DataFrame)

X = pendulum_output.time
Y = pendulum_output.pos

print(size(pendulum_output))