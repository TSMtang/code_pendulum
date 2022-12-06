using DataFrames
using CSV

data_pen = DataFrame(time = x,pos = y1)
CSV.write("C:\\Users\\Proteus\\Desktop\\RWTH\\WISE23\\Code_Pendulum\\CIE\\ProjectA\\code_pendulum\\pen.csv",data_pen)