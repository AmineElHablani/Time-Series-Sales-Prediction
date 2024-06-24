# Time-Series-Sales-Prediction

## Overview
This project focuses on forecasting sales for a superstore using various time series models and machine learning techniques. The project utilizes traditional models like ARIMA, SARIMA, autoARIMA, Prophet, and Neural Prophet, as well as machine learning regression models. To optimize the code and automate repetitive processes, Object-Oriented Programming (OOP) principles were employed.

## Features
- **Time Series Models:** Implementation of ARIMA, SARIMA, autoARIMA, Prophet, and Neural Prophet models.
- **Machine Learning Models:** Application of regression models for sales prediction.
- **Additional Forecasting Models:** Integration of models such as auto_arima, simple exponential smoothing, and theta algorithm from the Darts library.
- **OOP Optimization:** Code optimization through OOP principles to automate and streamline the forecasting process.
- **Custom Auto ARIMA Function:** Development of a custom function that automates ARIMA and SARIMA training, yielding better results than standard autoARIMA.
- **Custom Auto ARIMA Function:** Development of a custom function that automates the training of multiple ARIMA and SARIMA models. This function automatically checks for ACF and PACF, determines seasonality and stationarity, and differentiates the data if necessary (in case of non-stationarity). It has returned better results than the standard autoARIMA function.
