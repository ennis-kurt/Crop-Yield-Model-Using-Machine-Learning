# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Modelling Corn Production Yield
# <div class="span5 alert alert-info">
# 1. There are two main ways to predict the crop yield. The first one is from simple time series analysis of crop yield data, and building a time series model such as ARIMA. This method is straightforward, does not require any variable other than the yield itself and time as the single dimension. However, this method does not provide any physical inside for the problem and assumes that all the conditions relavent to crop production will be the same in the future. Despite the weakneses it can still provide a good starting point and can be useful to see how will the yield change in the future all the conditions stay the same as the past.
#
# 2. Second method would be building regression models to predict the crop yield from the actual physical parameters. This method is superior to time series analysis in terms of providing more actionable results, such as if we determine that the most important parameter is the rain amount during the growing season we could suggest irrigation to increase. 
#
# Here in this project I will use ARIMA mode and discuss the findings at the end.
#     </div>

# %%
import os
import cdsapi

import numpy as np
import pandas as pd
import pandas_profiling
import geopandas
import netCDF4
import xarray as xarr # pandas based library for 
            # labeled data with N-D tensors at each dimension
import salem

import matplotlib.pyplot as plt
# %matplotlib inline 
import cartopy
import cartopy.crs as ccrs
import seaborn as sns

# %% [markdown]
# ## 1. ARIMA Model

# %%
# Impoting the libraries required for this section
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

from sklearn.metrics import mean_squared_error

# %% [markdown]
# <div class="span5 alert alert-warning">We will use corn, grain yield data to build an ARIMA model. For this purpose, we do not need the agro-climate indicators.
# Let's remember how the corn data looks.
#     </div>

# %% tags=[]
crop_trend_plot("CORN, GRAIN", 'YIELD')

# %% [markdown]
# Let's start with only 'ILLINOIS'.

# %%
# getting the only corn - yield rows
corn_mask = df_crop_srv['Data Item'] == \
    "CORN, GRAIN - YIELD, MEASURED IN BU / ACRE"
df_corn = df_crop_srv[corn_mask].reset_index(drop=True)
# creating dataframe for soybens in Illinois 
df_corn_il = df_corn[df_corn.State == 'ILLINOIS'].sort_values(by='Year').reset_index(drop=True)
df_corn_il.info()

# %%
# Converting Year column to datetime object and setting as index
df_corn_il['Year'] = pd.to_datetime(df_corn_il['Year'], format='%Y')
df_corn_il.set_index('Year', inplace=True)

# %%
# Plotting corn yield for Illinois alone
# _ = df_corn_il['Value'].pct_change().plot(title="Corn Yield - ILLINOIS")
corn_yield = df_corn_il['Value']
_ = yield_soy.plot(title="Corn Yield - ILLINOIS")
# _ = df_corn_il.plot(x='Year', y='Value')

# %% [markdown]
# ### Model Identification
# Before we fit a model, we need to check if the data is stationary. There is abviously a strong trend starting from 1940s, but is it also a random walk? What does it take to make the model stationary?
# * Test the null hypothesis that the model is random walk with Dicky-Fuller Test.
# * Test the hypothesis that model is stationary with KPSS tests.
# * Make the model stationary taking difference of the values.
# * Plot the auto correlation function, and partial auto correlation function of the data to identify possible model order.
# * Build multiple ARIMA models and with different orders and find the best one in terms of AIC and BIC scores

# %%
adf = adfuller(corn_yield)[1]
kpss_ = kpss(corn_yield, nlags="auto")[1]
print(f"Dickey-Fuller test p-value is {round(adf,3)}")
print(f"KPSS test p-value is {round(kpss_, 3)}")

# %% [markdown]
# * Based on the Dickey-Fuller test we can not reject the null hypothesis which is the series has a unit root.
# * The null hypothesis of the KPSS test is the opposite, which is the process is trend stationary. Since the p-value is smaller than 0.05 we can reject the null hypothesis in favor of the alternative. 
#
# Thus both of the test suggest that the series is non-stationary. The easist way to get rid of the trend and make the data stationary is to take the lagged difference of the values. In most of the cases this will take care of non-stationarity. Let's try a differencing for a few lag.

# %% [markdown]
# The auto correlation plot tails off while the partial autocorrelation cuts off at lag 2. This suggest AR(2) model. However, we should check the acf and pacf plots after removing the trend in the next step. 

# %%
# from sklearn.linear_model import LinearRegression
# model = LinearRegression()
# x= np.linspace(1, len(yield_soy), num=len(yield_soy) ).reshape(-1,1)
# model.fit(x, yield_soy)
# trend = model.predict(x)
# _ = plt.plot(yield_soy.values,label='Yield')
# _ = plt.plot(trend, label='trend')
# _ = plt.legend()
# _ = plt.title('The Linear Trend')

# %%
# # Now removing the trend from the data
# y = yield_soy.values - trend
# y = pd.Series(y, index=yield_soy.index)
# _ = plt.plot(y)

# %% [markdown]
# Let's see whether removing the trend helped to make the data stationary or not.

# %%
adf = adfuller(y)[1]
kpss_ = kpss(y, nlags="auto")[1]
print(f"Dickey-Fuller test p-value is {round(adf,4)}")
print(f"KPSS test p-value is {round(kpss_, 4)}")

# %%
for i in range(4):
    if i == 0:
        y_diff = yield_soy
    elif i > 0:
        y_diff = yield_soy.diff(periods=i).dropna()
        
    print(f'p-value of randomness \
    for period={i} = {adfuller(y_diff)[1]}')
    
    print(f'p-value of stationarity \
    for period={i} = {kpss(y_diff, nlags="auto")[1]}')

# %% [markdown]
# #### Test Results
# **Dickey-Fuller Test:** The first order differencing is sufficient with p-value << 0.05
# **KPSS Test:** the first order difference is a weak stationary with p = 0.072. 
# The succesive orders make it worse. This is probably because of the variance that is changing with time. Let's see the data after differencing.

# %%
_ = yield_soy.diff().dropna().plot(title='Corn Yield - $1^{st}$ Order Difference')

# %% [markdown]
# Look's like the differencing took care of trend but the variance changes in time. The easiest way to make the variance constont is taking the log of the timeseries first

# %%
# Taking the first difference
ylog = np.log(yield_soy)
ylog_diff = ylog.diff().dropna()
# plotting y
_ = ylog_diff.plot(title = 
                   ' $ln {\ (Corn\ Yield)}$ -  $1^{st}$ Order Difference' )

# %%
# Let's make the stationarity tests again.
adf = adfuller(ylog_diff)[1]
kpss_ = kpss(ylog_diff, nlags="auto")[1]
print(f"Dickey-Fuller test p-value is {round(adf,4)}")
print(f"KPSS test p-value is {round(kpss_, 4)}")

# %% [markdown]
# This looks better in terms of variance and overall stationarity of the data. We will fit an ARIMA model now. Thus we actually do not need to take the difference but we should take the logarithm first.

# %%
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,6))

# Plot ACF and PACF
plot_acf(ylog_diff, lags=15, zero=False, ax=ax1)
plot_pacf(ylog_diff, lags=15, zero=False, ax=ax2)

# Show plot
plt.tight_layout()
plt.show()


# %% [markdown]
# * The auto correlation plot cuts off after the first lag 1.
# * PACF tails off. 
#
# <div class="span5 alert alert-info">
# An ARIMA model with the order (0,1,1) might fit for the time series.
#
# Recall the model choosing criteria based on ACF and PACF plots: </div>
#
# <table><tbody><tr><th></th><th>AR(p)</th><th>MA(q)</th><th>ARMA(p,q)</th></tr><tr><td>ACF</td><td>Tails off</td><td>Cuts off after lag q</td><td>Tails off</td></tr><tr><td>PACF</td><td>Cuts off after lag p</td><td>Tails off</td><td>Tails off</td></tr></tbody></table>
#

# %% [markdown]
# ### Model Selection
# Three criteria will be used for model selection.
# 1. MSE error based on timestep-wise comparison between test data and one-step prediction ARIMA model.
# 2. Akaike information criteria
# 3. Bayesian information criteria

# %%
# Import mean_squared_error and ARIMA

# Make a function called evaluate_arima_model to find the MSE of a single ARIMA model 
def evaluate_arima_model(data, arima_order):
    # Needs to be an integer because it is later used as an index.
    # Use int()
    split = int(len(data) * 0.8) 
    # Make train and test variables, with 'train, test'
    train, test = data[0:split], data[split:len(data)]
    past=[x for x in train]
    # make predictions
    predictions = list()
    for i in range(len(test)):#timestep-wise comparison between test data and one-step prediction ARIMA model. 
        model = ARIMA(past, order=arima_order)
        model_fit = model.fit()
        future = model_fit.forecast()[0]
        predictions.append(future)
        past.append(test[i])
    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    # Return the error
    return error


# %%
# Function to evaluate different ARIMA models with several different p, d, and q values.
def evaluate_models(dataset, p_values, d_values, q_values):
    score_dict = {'order':[], 'mse':[]}
    best_score, best_cfg = float("inf"), None
    # Iterate through p_values
    for p in p_values:
        # Iterate through d_values
        for d in d_values:
            # Iterate through q_values
            for q in q_values:
                # p, d, q iterator variables in that order
                order = (p, d, q)
                try:
                    # Make a variable called mse for the Mean squared error
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    score_dict['order'].append(order)
                    score_dict['mse'].append(mse)
                    print(f'ARIMA{order} MSE={round(mse, 4)}')
                except:
                    continue
    return pd.DataFrame(score_dict), print('Best ARIMA%s MSE=%.4f' % (best_cfg, best_score))


# %%
# Now, we choose a couple of values to try for each parameter.
# let's try up to 3 for each parameter
p_values = [i for i in range(3)] # from pacf plot the best p might be 3
d_values = [i for i in range(3)] # p-val of kpss lower but still close to 0.05. Let's try up to d=2
q_values = [i for i in range(3)] # Most likely we have model with MA order 0 since pacf plot cuts off

# %% [markdown]
# Now fitting corn_yield. Note that, this is the initial time series in which the differencing has not been done.

# %%
# Finally, we can find the optimum ARIMA model for our data.
import warnings
warnings.filterwarnings("ignore")
scores = evaluate_models(ylog, p_values, d_values, q_values)

# %% [markdown]
# The best ARIMA model based on MSE is ARIMA(0,1,1). Let's compare the models based on AIC and BIC scores to be more confident about our final model.

# %%
# Create empty list to store search results
order_aic_bic=[]

# Loop over p values from 0-2
for p in range(3):
  # Loop over d values from 0-2
    for d in range(3):    
    # Loop over q values from 0-2
        for q in range(3):
            # create and fit ARIMA(p,d,q) model
            try:
                model = ARIMA(ylog, order=(p, d, q))
                results = model.fit(disp=0)

                # Append order and results tuple
                order_aic_bic.append((p, d, q, results.aic, results.bic))
            except:
                continue

# %%
# Construct DataFrame from order_aic_bic
order_df = pd.DataFrame(order_aic_bic, 
                        columns=['p', 'd', 'q', 'AIC', 'BIC'])

# Print order_df in order of increasing AIC
order_df.sort_values('AIC').reset_index(drop=True).head()

# %%
# Print order_df in order of increasing BIC
order_df.sort_values('BIC').reset_index(drop=True).head()

# %% [markdown]
# All three criteria, (BIC, AIC and MSE) point to same model: ARIMA(0,1,1), just like we anticipated from the ACF and PACF plots.

# %% [markdown]
# All three criteria, (BIC, AIC and MSE) points to same model: ARIMA(0,1,1), just like we anticipated from the ACF and PACF plots. 
#
# * The best model based on BIC is: ARIMA(0,1,1)
# * The best model based on AIC ARIMA(2, 1, 2), but ARIMA(0, 1, 1) score is very close to best score
# * The Best models based on the MSE are ARIMA(2,1,2), but ARIMA(0,1,1) is the simplest model with one of the lowest score
# Since the ARIMA(0,1,1) is a much simplier model than ARIMA(2,1,2) and still performance almost as well as the best model based on AIC and MSE criteria, I will fit the series with ARIMA(0,1,1)

# %%
arima = ARIMA(ylog,order=(0,1,1))
model = arima.fit()
forecast = model.forecast(25)
y_pred = model.predict()

# %%
model.summary()

# %% [markdown]
# <p>Here is a reminder of the tests in the model summary:</p>
#
# <table>
#   <tbody><tr>
#     <th>Test</th>
#     <th>Null hypothesis</th>
#     <th>P-value name</th>
#   </tr>
#   <tr>
#     <td>Ljung-Box</td>
#     <td>There are no correlations in the residual<br></td>
#     <td>Prob(Q)</td>
#   </tr>
#   <tr>
#     <td>Jarque-Bera</td>
#     <td>The residuals are normally distributed</td>
#     <td>Prob(JB)</td>
#   </tr>
# </tbody></table>

# %% [markdown]
# PRob(Q) = 0.13 > 0.05. We should reject the null hypothesis and deduce that there are correlations in the residuals. Moreover, Prob(JB) < 0.05 i.e. residuals not normally distribute based on the Jarque-Bera test. 

# %% [markdown]
# ### Model Diagnostics

# %%
_ = fitted.plot_predict()

# %%
import statsmodels.api as sm
import scipy.stats as stats
# Plot residual errors
residuals = pd.DataFrame(model.resid)
fig, ax = plt.subplots(2,2,figsize=(12,8))
# residuals.plot.scatter(x=residuals.index, y=residuals.values, title="Residuals", ax=ax[0], linestyle=None, marker='.')
ax[0,0].scatter(x=residuals.index, y=residuals.values, marker='.')
ax[0,0].plot(residuals.index, np.zeros(len(residuals)), 'k--')
plot_acf(residuals, ax=ax[0,1], zero=False)
# residuals.plot(kind='kde', title='Density', ax=ax[1,0])
sns.kdeplot(residuals.values.reshape(-1,), ax=ax[1,0])
ax[1,0].hist(residuals,density=True)
# sm.qqplot(residuals, line='45',ax=ax[1,1])
stats.probplot(residuals.values.reshape(-1,), dist="norm", plot=plt)
plt.tight_layout()
plt.show()

# %%
from scipy.stats import shapiro
stat, p = shapiro(residuals)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print("Data follows Normal Distribution")
else:
    print("Data does not follow Normal Distribution")

# %%
from scipy.stats import anderson
result = anderson(residuals.values.reshape(-1,))
print('stat=%.3f' % (result.statistic))
for i in range(len(result.critical_values)):
 sl, cv = result.significance_level[i], result.critical_values[i]
 if result.statistic < cv:
  print('Data follows Normal at the %.1f%% level' % (sl))
 else:
  print('Data does not follows Normal at the %.1f%% level' % (sl))

# %% [markdown]
# <div class="span5 alert alert-warning">
# Model have some outliers, the residuals are not normally distributed. The only good news is, there is no correlation in the residuals.
#     </div>

# %% [markdown]
# Let's see the model with the un-modified yield data, plus with a dynamic forecast.

# %%
# Create Training and Test
train = ylog[:119]
test = ylog[119:]
print(f'Test percent: {100*round(len(test)/len(train), 2)}')


# %%
# Build Model
# model = ARIMA(train, order=(3,2,1))    
def plot_dynamic_pred(dt, dtrain, dtest, order):
    model = ARIMA(dtrain, order=order)  
    fitted = model.fit(disp=-1)  
    ypred = model.predict(dtrain, dynamic=True)
    ypred = pd.Series(ypred,index=dtrain.index[order[1]:])
    forecast_period= len(test)+25
    # Forecast
    fc, se, conf = fitted.forecast(forecast_period, alpha=0.05)  # 95% conf

    # Make as pandas series

    date_range = pd.date_range(dt.index[-len(test)], periods = forecast_period, 
                  freq='Y').strftime("%Y-%m-%d").tolist()
    date_range = pd.to_datetime(date_range)

    fc_series = pd.Series(fc, index=date_range)
    lower_series = pd.Series(conf[:, 0], index=date_range)
    upper_series = pd.Series(conf[:, 1], index=date_range)

    # Plot
    _=plt.figure(figsize=(12,5), dpi=100)
    _ = plt.plot(dtrain, label='training')
    # _ = fitted.plot_predict()
    _= plt.plot(dtest, label='test')
    _= plt.plot(fc_series, label='forecast', c='k')
    _= plt.fill_between(lower_series.index, lower_series, upper_series, 
                     color='k', alpha=.15)

    _=plt.title('Dynamic Forecast vs Actual Values')
    _=plt.legend(loc='upper left', fontsize=8)
    # plt.show()
plot_dynamic_pred(ylog,train, test, (0,1,1))

# %% [markdown]
# <div class="span5 alert alert-info">
#     Finally let's try the original data without taking logarithm as we did before to fix the variance change in time. In the previous model, we had a trouble finding the best model order. We chose ARIMA(0,1,1), but the residuals were not normal. Whereas any more complicated models actually make worse of both the normallity of the residuals and other criteria like mse. </div>

# %% [markdown]
# This time let's use sklearns' SARIMAX method, which is equivalent its ARIMA model but have some more features.

# %%
# Let's test mse for ARIMA models with original data
y = corn_yield
import warnings
warnings.filterwarnings("ignore")
scores = evaluate_models(y, p_values, d_values, q_values)

# %%
# Create empty list to store search results
order_aic_bic=[]

# Loop over p values from 0-2
for p in range(3):
  # Loop over d values from 0-2
    for d in range(3):    
    # Loop over q values from 0-2
        for q in range(3):
            # create and fit ARIMA(p,d,q) model
            try:
                model = ARIMA(y, order=(p, d, q))
                results = model.fit(disp=0)

                # Append order and results tuple
                order_aic_bic.append((p, d, q, results.aic, results.bic))
            except:
                continue

# %%
# Construct DataFrame from order_aic_bic
order_df = pd.DataFrame(order_aic_bic, 
                        columns=['p', 'd', 'q', 'AIC', 'BIC'])

# Print order_df in order of increasing AIC
order_df.sort_values('AIC').reset_index(drop=True).head()

# %%
order_df.sort_values('BIC').reset_index(drop=True).head()

# %% [markdown]
# All three criteria we used point to same model, again, but this time the order of the best model is (0,2,2)

# %%
# Calling SARIMAX and fitting to original data
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(0, 2, 2))

results = mod.fit()

# one-step ahead prediction
pred = results.get_prediction(start=pd.to_datetime('1870-01-01'), dynamic=False)
pred_ci = pred.conf_int()

# Dynamic Prediction
pred_dynamic = results.get_prediction(start=pd.to_datetime('1945-01-01'), dynamic=True, full_results=True)
pred_dynamic_ci = pred_dynamic.conf_int()

# Get forecast 500 steps ahead in future
pred_uc = results.get_forecast(steps=50)

# Get confidence intervals of forecasts
pred_uc_ci = pred_uc.conf_int()

# %%
fig, ax = plt.subplots(figsize=(10,5))
ax = corn_yield.plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
pred_uc.predicted_mean.plot(ax=ax, label='Forecast', c='k')
ax.fill_between(pred_uc_ci.index,
                pred_uc_ci.iloc[:, 0],
                pred_uc_ci.iloc[:, 1], color='k', alpha=.25)

ax.set_xlabel('Date')
ax.set_ylabel('Corn, Grain - Yield')
plt.legend()

plt.show()
 


# %% [markdown]
# #### Plot Dynamic prediction

# %%
# Create Training and Test and plot dynamic prediction
train = y[:119]
test = y[119:]
plot_dynamic_pred(y,train, test, (0,2,2))

# %% [markdown]
# ### Conclusion For ARIMA Models
#
# <div class="span5 alert alert-info">
#     Arima models for the corn, grain production for ILLINOIS, can be considered as having limited prediction capability. I tried two models, one with the original data and the other one with the natural logarithm of the actual values to make the data more stationary by cancelling the temporal variance change. However, both of the model has suffered from non-normal residual distriution at the end. Both of the model have residuals without autocorrelation at any lag. 
#     
# Overall, I would not suggest using ARIMA model for corn, grain yield. Although here I showed analysis for ILLINOIS, I made the same analysis for a few other states, none of which showed any promises for an ARMA model. Also not shown here is the soy bean yields, which have very similar results. 
#
# Perhaps,the better way to model crop yield would be using the agro-climatic indicators as features and building a regression model. In the next chapter I will do this. However, this time the major limitation is the lack of sufficiently long time series data for agro-climatic indicators, which starts at 1980s. The major challenge would be cross-validation and testing the model.
#     </div>
