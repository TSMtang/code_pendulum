using DataFrames
using CSV
using Flux

pendulum_output = CSV.read("C:\\Users\\Proteus\\Desktop\\RWTH\\WISE23\\Code_Pendulum\\CIE\\ProjectA\\code_pendulum\\pendulum.csv", DataFrame)

print(pendulum_output.time)