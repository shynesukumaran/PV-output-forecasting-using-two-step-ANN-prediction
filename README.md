# PV-output-forecasting-using-two-step-ANN-prediction

The aim is to predict the future output power values of a PV system. The historical data of weather parameters- Air temperature, Wind speed, Solar radiation- and PV output power for a specific period is available.

At first, an ANN model is trained on the weather parameters and the PV output power values. However, to get the future prediction of PV output power values, weather forecast is required which will give the future values of Air temperature, Wind speed and Solar radiation. Unfortunately, solar radiation forecast values are not available with the available free API calls to weather websites. 

To solve this problem, that is, to get the future values of solar radiation, another ANN is trained to predict the values of solar radiation based on other weather parameters available through API. Once this ANN predicts the solar radiation values, it is combined with the forecast values of air temperature and wind speed obtained through API call. Then it is given to the ANN model already trained for prediction using input parameters - wind speed, air temperature and solar radiation.
