using CSV
using DataFrames
using Plots
using StateSpaceModels

# load data
data = readdlm("data.txt", '\t', Float64, '\n')
t = data[:,1]
phi = data[:,2]
phi_dot = data[:,3]

airp = CSV.File(StateSpaceModels.AIR_PASSENGERS) |> DataFrame
log_air_passengers = log.(airp.passengers)
log_air_passengers = phi[1:1000]
steps_ahead = 1000

# SARIMA
#model_sarima = SARIMA(log_air_passengers; order = (0, 1, 1), seasonal_order = (0, 1, 1, 12))
#fit!(model_sarima)
#forec_sarima = forecast(model_sarima, steps_ahead)

# Unobserved Components
#model_uc = UnobservedComponents(log_air_passengers; trend = "local linear trend", seasonal = "stochastic 12")
#fit!(model_uc)
#forec_uc = forecast(model_uc, steps_ahead)

# Exponential Smoothing
model_ets = ExponentialSmoothing(log_air_passengers; trend = true, seasonal = 12)
fit!(model_ets)
forec_ets = forecast(model_ets, steps_ahead)

# Naive model
model_naive = SeasonalNaive(log_air_passengers, 12)
fit!(model_naive)
forec_naive = forecast(model_naive, steps_ahead)

#plt_sarima = plot(model_sarima, forec_sarima; title = "SARIMA", label = "");
#plt_uc = plot(model_uc, forec_uc; title = "Unobserved components", label = "");
plt_ets = plot(model_ets, forec_ets; title = "Exponential smoothing", label = "");
plt_naive = plot(model_ets, forec_naive; title = "Seasonal Naive", label = "");

plot(plt_sarima, plt_uc, plt_ets, plt_naive; layout = (2, 2), size = (500, 500))
