import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import LinearRegression

import wrangle

def reshape_series(series):
    """For getting a 2D array from a pandas series. Useful for univariate models."""
    x = np.array(series)
    x = x.reshape((x.shape[0], 1))
    return x

def train_baseline_match_len(predictions):
    ybar = train.tax.mean()
    baseline = np.full_like(predictions, ybar)
    return baseline

train, test, validate = wrangle.wrangle_zillow()

x1 = reshape_series(train.structure_sqft)
y1 = train.tax_structure

x2 = reshape_series(train.age)
y2 = y1
poly = PolynomialFeatures(3, include_bias=False, interaction_only=False)
poly.fit(x2)
x2 = poly.transform(x2)

x3 = reshape_series(train.tax_structure)
y3 = train.tax


model1 = LinearRegression(positive=False)
model1.fit(x1, y1)

model2 = LinearRegression(positive=False)
model2.fit(x2, y2)

model3 = LinearRegression(positive=False)
model3.fit(x3,y3)

def ensemble_predict(xdf, m1=model1, m2=model2, m3=model3):
    p1 = m1.predict(reshape_series(xdf.structure_sqft))
    p2 = m2.predict(reshape_series(xdf.age))
    p1p2 = (p1+p2)/2
    p1p2 = reshape_series(p1p2)
    p3 = m3.predict(p1p2)
    return p3

