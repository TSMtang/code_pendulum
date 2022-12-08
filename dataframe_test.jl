using CSV
using DataFrames
#import CSV.write

#x_test = [1, 2, 3, 4, 5]
#y1_test = [6, 7, 8, 9, 10]


data_pendulum = DataFrame(time_input = [1, 2, 3, 4, 5], position_output = [6, 7, 8, 9, 10])
#path = "/Users/franzoldopp/Projekte/CIE/code_pendulum_franz/code_pendulum\\test_csv.csv"

CSV.write("\\Users\\franzoldopp\\Projekte\\CIE\\code_pendulum_franz\\code_pendulum\\test_csv.csv", data_pendulum)


#XLSX.writetable(path, collect(eachcol(data_pendulum)), names(data_pendulum))