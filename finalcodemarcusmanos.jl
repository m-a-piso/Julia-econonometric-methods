# disclaimer: this code is going to save plots and a .csv file in your working directory when run!!!
# authors: Marcus-Adrian Piso (i6314615), Manuel Hadjimarkos (i6341105)


# load required packages
using MarketData       # for yahoo finance data
using DataFrames       # dataframes for table format
using TimeSeries       # needed to get dates
using Dates            # date types
using Statistics       # for mean std etc
using StatsBase        # skewness and kurtosis
using HypothesisTests  # jarque bera test
using GLM              # for lm function
using CSV              # to save the csv
using LinearAlgebra    # for matrix operations

# Define explicit date range
start_date = Date("2015-01-01")
end_date = Date("2025-01-01")

# define stock tickers
tickers = Dict(
    "AMZN" => :AMZN,
    "PYPL" => :PYPL,
    "NKE"  => :NKE,
    "PEP"  => :PEP,
    "NFLX" => :NFLX
)

# index symbol for nasdaq 100
index_sym = "^NDX"

# show that download is starting
println("getting data...")

# make dict to store price and return data
stock_data = Dict{String,DataFrame}()

# loop over each stock to get price data
for (label, symbol) in tickers
    ts = yahoo(symbol)
    dts = TimeSeries.timestamp(ts)
    close_px = values(ts[:Close])

    # make df with date and price
    df = DataFrame(Date = dts, Close = close_px)

    # explicitly filter by date
    df = filter(row -> start_date <= row.Date <= end_date, df)

    # compute log returns
    n = length(df.Close)
    log_ret = Vector{Float64}(undef, n)
    log_ret[1] = NaN  # no return for first obs

    for i in 2:n
        log_ret[i] = log(df.Close[i] / df.Close[i-1])
    end

    df[!, :Returns] = log_ret

    # remove first row
    stock_data[label] = df[2:end, :]
end

# get index data
idx_data = yahoo(index_sym)
idx_dts = TimeSeries.timestamp(idx_data)
idx_px = values(idx_data[:Close])

# make df for index
index_df = DataFrame(Date = idx_dts, Index_Close = idx_px)

# explicitly filter by date
index_df = filter(row -> start_date <= row.Date <= end_date, index_df)

# log returns for index
n = length(index_df.Index_Close)
idx_ret = Vector{Float64}(undef, n)
idx_ret[1] = NaN
for i in 2:n
    idx_ret[i] = log(index_df.Index_Close[i] / index_df.Index_Close[i-1])
end

index_df[!, :Returns] = idx_ret
index_df = index_df[2:end, :]

println("done computing returns")

# plots for each stock
using Plots          # need this for plotting
using Distributions  # for normal curve on histogram

# function to make plots for one asset
function make_plots(name, df, price_col = :Close)
    # price over time
    p1 = plot(df.Date, df[:, price_col],
              title = "$name price",
              ylabel = "price",
              label = false,
              legend = false,
              linewidth = 1.5,
              color = :blue)

    # log returns over time
    p2 = plot(df.Date, df.Returns,
              title = "$name log returns",
              ylabel = "return",
              label = false,
              legend = false,
              linewidth = 1,
              color = :red)

    # horizontal line at 0
    hline!(p2, [0], linestyle = :dash, color = :black, alpha = 0.5, label = false)

    # histogram of returns with normal curve
    p3 = histogram(df.Returns,
                   title = "$name returns distribution",
                   xlabel = "return",
                   ylabel = "frequency",
                   bins = 50,
                   normalize = true,
                   alpha = 0.7,
                   color = :lightblue,
                   label = "returns")

    # compute normal curve
    μ = mean(df.Returns)
    σ = std(df.Returns)
    xvals = range(μ - 4σ, μ + 4σ, length = 100)
    yvals = [pdf(Normal(μ, σ), x) for x in xvals]

    # plot normal curve on top
    plot!(p3, xvals, yvals, line = (2, :dash, :red), label = "normal")

    # combine all 3 plots
    return plot(p1, p2, p3, layout = (3,1), size = (900, 800))
end

# check for folder
if !isdir("plots")
    mkdir("plots")
end

# loop over each stock and save plots
for (name, df) in stock_data
    fig = make_plots(name, df)
    savefig(fig, "plots/$(name)_plots.png")
    display(fig)
end

# do the same for the index
index_fig = make_plots("NASDAQ-100", index_df, :Index_Close)
savefig(index_fig, "plots/NASDAQ-100_plots.png")
display(index_fig)

println("plots saved in 'plots' folder")

# print summary stats and normality test
println("\n===== summary statistics =====")

# go through each stock
for (name, df) in stock_data
    println("\n$name returns:")
    r = df.Returns

    # jarque bera test for normality
    jb = JarqueBeraTest(r)

    # basic stats
    println(describe(r))
    println("Skewness: $(skewness(r))")
    println("Kurtosis: $(kurtosis(r))")
    println("Jarque-Bera p-value: ", round(pvalue(jb), digits=12))
end

# now for the index
println("\nNASDAQ-100 returns:")
r = index_df.Returns
jb = JarqueBeraTest(r)
println(describe(r))
println("Skewness: $(skewness(r))")
println("Kurtosis: $(kurtosis(r))")
println("Jarque-Bera p-value: ", round(pvalue(jb), digits=12))


# make equal weighted portfolio from the 5 stocks
# average return each day

dates = stock_data["AMZN"].Date
port_ret = zeros(length(dates))

for i in 1:length(dates)
    d = dates[i]
    total = 0.0
    count = 0

    # go through each stock and add returns for that date
    for (name, df) in stock_data
        idx = findfirst(x -> x == d, df.Date)
        if idx !== nothing
            total += df.Returns[idx]
            count += 1
        end
    end

    # get mean return for the day
    if count > 0
        port_ret[i] = total / count
    else
        port_ret[i] = NaN
    end
end

# save to dataframe
portfolio_df = DataFrame(Date = dates, Portfolio = port_ret)

# print stats for portfolio
println("\n===== portfolio statistics =====")
r = filter(!isnan, portfolio_df.Portfolio)
jb = JarqueBeraTest(r)
println(describe(r))
println("  skewness: $(skewness(r))")
println("  kurtosis: $(kurtosis(r))")
println("  Jarque-Bera p-value: $(pvalue(jb))")

# compare stats across all stocks + index + portfolio

println("\n===== comparison =====")

# mean returns
println("means:")
for (name, df) in stock_data
    println("  $name: $(mean(df.Returns))")
end
println("  NASDAQ-100: $(mean(index_df.Returns))")
println("  portfolio: $(mean(filter(!isnan, portfolio_df.Portfolio)))")

# standard deviations
println("\nvolatilities (std dev):")
for (name, df) in stock_data
    println("  $name: $(std(df.Returns))")
end
println("  NASDAQ-100: $(std(index_df.Returns))")
println("  portfolio: $(std(filter(!isnan, portfolio_df.Portfolio)))")

# p-values from jarque bera
println("\njarque-bera test p-values:")
for (name, df) in stock_data
    println("  $name: $(pvalue(JarqueBeraTest(df.Returns)))")
end
println("  NASDAQ-100: $(pvalue(JarqueBeraTest(index_df.Returns)))")
println("  portfolio: $(pvalue(JarqueBeraTest(filter(!isnan, portfolio_df.Portfolio))))")

# save summary stats to csv for later use
stats_data = DataFrame(
    Asset = String[],
    Mean = Float64[],
    Median = Float64[],
    Variance = Float64[],
    Skewness = Float64[],
    Kurtosis = Float64[],
    JB_pvalue = Float64[]
)

# fill in stats for each stock
for (name, df) in stock_data
    r = df.Returns
    jb = JarqueBeraTest(r)
    push!(stats_data, [
        name,
        mean(r),
        median(r),
        var(r),
        skewness(r),
        kurtosis(r),
        pvalue(jb)
    ])
end

# now for the index
r = index_df.Returns
jb = JarqueBeraTest(r)
push!(stats_data, [
    "NASDAQ-100",
    mean(r),
    median(r),
    var(r),
    skewness(r),
    kurtosis(r),
    pvalue(jb)
])

# and the portfolio
r = filter(!isnan, portfolio_df.Portfolio)
jb = JarqueBeraTest(r)
push!(stats_data, [
    "portfolio",
    mean(r),
    median(r),
    var(r),
    skewness(r),
    kurtosis(r),
    pvalue(jb)
])

# write to file
CSV.write("summary_statistics.csv", stats_data)

# tests for autocorrelation and volatility clustering

# breusch godfrey test for autocorr
function bg_test(r)
    n = length(r)
    
    # step 1: regress return on const
    X = ones(n - 1, 1)
    y = r[2:end]
    model = lm(X, y)
    res = GLM.residuals(model)

    # step 2: regress res on lagged res
    X_lag = [ones(n - 2, 1) res[1:end-1]]
    y_lag = res[2:end]
    model_lag = lm(X_lag, y_lag)

    # step 3: lm stat
    R² = r2(model_lag)
    LM = (n - 1) * R²

    # p-value from chi2(1)
    pval = 1 - cdf(Chisq(1), LM)

    return (LM = LM, pval = pval)
end

# run test for all stocks
println("\n===== breusch-godfrey test for autocorrelation =====")
for (name, df) in stock_data
    r = df.Returns[2:end]
    out = bg_test(r)
    println("$name: LM = $(out.LM), p = $(out.pval)")
end

# index
r = index_df.Returns[2:end]
out = bg_test(r)
println("NASDAQ-100: LM = $(out.LM), p = $(out.pval)")

# portfolio
r = filter(!isnan, portfolio_df.Portfolio)[2:end]
out = bg_test(r)
println("portfolio: LM = $(out.LM), p = $(out.pval)")

# arch lm test for heteroskedasticity
function arch_test(r)
    n = length(r)
    
    # step 1: basic model
    X = ones(n - 1, 1)
    y = r[2:end]
    model = lm(X, y)
    res = GLM.residuals(model)

    # step 2: squared residuals on lag
    X_lag = [ones(n - 2, 1) res[1:end-1].^2]
    y_lag = res[2:end].^2
    model_lag = lm(X_lag, y_lag)

    # stat and pval
    R² = r2(model_lag)
    LM = (n - 1) * R²
    pval = 1 - cdf(Chisq(1), LM)

    return (LM = LM, pval = pval)
end

# run test
println("\n===== arch lm test for heteroskedasticity =====")
for (name, df) in stock_data
    r = df.Returns[2:end]
    out = arch_test(r)
    println("$name: LM = $(out.LM), p = $(out.pval)")
end

# index
r = index_df.Returns[2:end]
out = arch_test(r)
println("NASDAQ-100: LM = $(out.LM), p = $(out.pval)")

# portfolio
r = filter(!isnan, portfolio_df.Portfolio)[2:end]
out = arch_test(r)
println("portfolio: LM = $(out.LM), p = $(out.pval)")



# ar(1) regression with ols and robust (white) standard errors

function ols_with_hc(X, y)
    model = lm(X, y)
    n, k = size(X)
    res = GLM.residuals(model)

    # usual std errors
    se_ols = sqrt.(diag(vcov(model)))

    # white heteroskedasticity robust std errors
    X_sq = X .* res.^2
    vcov_hc = inv(X'X) * (X' * X_sq) * inv(X'X)
    se_hc = sqrt.(diag(vcov_hc))

    return (coeff = coef(model), ols_se = se_ols, robust_se = se_hc)
end

# run regression for each asset
println("\n===== regressions with ols and robust std errors =====")
for (name, df) in stock_data
    r = df.Returns
    X = ones(length(r) - 1, 2)
    X[:, 2] = r[1:end-1]
    y = r[2:end]

    out = ols_with_hc(X, y)

    println("\n$name:")
    println("intercept: $(out.coeff[1])")
    println("lagged return coef: $(out.coeff[2])")
    println("ols std errors: intercept = $(out.ols_se[1]), lagged = $(out.ols_se[2])")
    println("robust std errors: intercept = $(out.robust_se[1]), lagged = $(out.robust_se[2])")
end

# now for index
r = index_df.Returns
X = ones(length(r) - 1, 2)
X[:, 2] = r[1:end-1]
y = r[2:end]

out = ols_with_hc(X, y)

println("\nNASDAQ-100:")
println("intercept: $(out.coeff[1])")
println("lagged return coef: $(out.coeff[2])")
println("ols std errors: intercept = $(out.ols_se[1]), lagged = $(out.ols_se[2])")
println("robust std errors: intercept = $(out.robust_se[1]), lagged = $(out.robust_se[2])")


# rolling forecast for returns and volatility, ar(1) & arch(1) structure, plots

function forecast_returns(r, window)
    n = length(r)
    preds = zeros(n - window)

    for i in 1:(n - window)
        X = ones(window, 2)
        X[:, 2] = r[i:(i + window - 1)]
        y = r[(i + 1):(i + window)]

        model = lm(X, y)
        preds[i] = coef(model)[1] + coef(model)[2] * r[i + window - 1]
    end

    return preds
end

function forecast_volatility(r, window)
    n = length(r)
    preds = zeros(n - window)

    for i in 1:(n - window)
        sq_r = r[i:(i + window - 1)].^2
        X = ones(window - 1, 2)
        X[:, 2] = sq_r[1:end-1]
        y = sq_r[2:end]

        model = lm(X, y)
        preds[i] = abs(coef(model)[1] + coef(model)[2] * sq_r[end])
    end

    return sqrt.(preds)
end

# set window size (assume 252 trading days)
window = 252

# forecast and plot for each stock
for (name, df) in stock_data
    r = df.Returns

    ret_pred = forecast_returns(r, window)
    vol_pred = forecast_volatility(r, window)

    # return plot
    ret_plot = plot(df.Date[(window+1):end], ret_pred,
                    title = "$name return forecast",
                    xlabel = "date",
                    ylabel = "forecasted return",
                    label = "forecast")

    savefig(ret_plot, "plots/$(name)_return_forecast.png")
    display(ret_plot)

    # volatility plot
    vol_plot = plot(df.Date[(window+1):end], vol_pred,
                    title = "$name volatility forecast",
                    xlabel = "date",
                    ylabel = "forecasted volatility",
                    label = "forecast")

    savefig(vol_plot, "plots/$(name)_volatility_forecast.png")
    display(vol_plot)
end

# root mean square error for return forecast
function rmse(actual, pred)
    return sqrt(mean((actual - pred).^2))
end

println("\n===== return forecast rmse =====")
for (name, df) in stock_data
    r = df.Returns
    pred = forecast_returns(r, window)
    actual = r[(window+1):end]
    err = rmse(actual, pred)
    println("$name: $err")
end

# index rmse
r = index_df.Returns
pred = forecast_returns(r, window)
actual = r[(window+1):end]
err = rmse(actual, pred)
println("NASDAQ-100: $err")


# estimate capm model for each asset
# R_i = alpha + beta * R_m + error

function capm_model(stock_r, market_r)
    n = min(length(stock_r), length(market_r))
    stock_r = stock_r[1:n]
    market_r = market_r[1:n]

    X = [ones(n) market_r]
    y = stock_r

    model = lm(X, y)
    alpha, beta = coef(model)

    # normal std errors
    vcov_ols = vcov(model)
    se_ols = sqrt.(diag(vcov_ols))
    ci_ols = [(alpha - 1.96 * se_ols[1], alpha + 1.96 * se_ols[1]),
              (beta - 1.96 * se_ols[2], beta + 1.96 * se_ols[2])]

    # white robust std errors
    res = GLM.residuals(model)
    X_sq = X .* res.^2
    vcov_hc = inv(X'X) * (X' * X_sq) * inv(X'X)
    se_hc = sqrt.(diag(vcov_hc))
    ci_hc = [(alpha - 1.96 * se_hc[1], alpha + 1.96 * se_hc[1]),
             (beta - 1.96 * se_hc[2], beta + 1.96 * se_hc[2])]

    return (alpha = alpha, beta = beta,
            ols_ci = ci_ols, hc_ci = ci_hc)
end

# run capm for each stock
println("\n===== capm regression results =====")
for (name, df) in stock_data
    stock_r = df.Returns
    market_r = index_df.Returns

    out = capm_model(stock_r, market_r)

    println("\n$name:")
    println("alpha: $(out.alpha)")
    println("beta: $(out.beta)")
    println("ols 95% ci for alpha: $(out.ols_ci[1])")
    println("ols 95% ci for beta: $(out.ols_ci[2])")
    println("robust 95% ci for alpha: $(out.hc_ci[1])")
    println("robust 95% ci for beta: $(out.hc_ci[2])")
end