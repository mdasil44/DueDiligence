# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

import os
import pandas as pd
from iexfinance.stocks import Stock
from iexfinance.stocks import get_historical_data
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt

os.environ['IEX_API_VERSION'] = 'iexcloud-sandbox'
os.environ['IEX_TOKEN'] = 'Tpk_c8897b1c65f241558f5d4d5477241552'

yrs = 5
startDate = datetime.now() - timedelta(days=yrs*365)
currDate = datetime.now()

stock = get_historical_data("AAPL", start=startDate, end=currDate, output_format='pandas')
#plt.plot(stock.index, stock.values)
#plt.show()
x = pd.to_datetime(stock.index).astype(np.int64)

Xtrain, Xtest, ytrain, ytest = train_test_split(x.values.reshape(-1, 1), stock.close.values, test_size = 0.25 , random_state = 0)

# Training 3 types of models using the stonk values

# Regressor using ridge model
RidgeModel = Pipeline([
    ('standardize', StandardScaler(with_mean=True, with_std=True)),
    ('poly', PolynomialFeatures(10, include_bias=0)),
    ('reg', Ridge(alpha=0.5, fit_intercept=True))
])
RidgeR2Cross = cross_val_score(RidgeModel, Xtrain, ytrain, cv=10, scoring='r2')
print(f"RidgeModel: {RidgeR2Cross.mean()}")

# Regressor using RandomForest
RF2 = RandomForestRegressor(n_estimators = 500, max_features = 1) #default max features is sqrt(vars)
RF2.fit(Xtrain, ytrain)
RandomForestAccuracy = RF2.score(Xtest, ytest)
print(f"Random Forest Accuracy: {RandomForestAccuracy}")

# Regressor using XGBoost tree
XGBoost = XGBRegressor(max_depth = 3, learning_rate = 0.1, n_estimators = 500, booster = 'gbtree', gamma=0.001)
XGBoost.fit(Xtrain, ytrain)
XGBoostAccuracy = XGBoost.score(Xtest, ytest)
print(f"XGBoostAcc: {XGBoostAccuracy}")

ridge = RidgeModel.fit(Xtrain, ytrain)
y_pred = ridge.predict(x.values.reshape(-1, 1))
y_true = stock.close.values
plt.plot(stock.index, y_pred, 'r')
plt.plot(stock.index, y_true)
plt.show()




