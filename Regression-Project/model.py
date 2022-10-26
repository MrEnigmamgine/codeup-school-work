import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import wrangle

from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linear_model import LinearRegression


def reshape_series(series):
    """For getting a 2D array from a pandas series. Useful for univariate models."""
    x = np.array(series)
    x = x.reshape((x.shape[0], 1))
    return x

def reshape_array(x):
    """For getting a 2D array from a 1D array. Useful for univariate models."""
    x = x.reshape((x.shape[0], 1))
    return x

def train_baseline_match_len(predictions):
    ybar = train.tax.mean()
    baseline = np.full_like(predictions, ybar)
    return baseline

train, test, validate = wrangle.wrangle_zillow()

x1 = reshape_series(train.structure_sqft)
x2 = reshape_series(train.tax_structure)
y1 = train.tax_structure
y2 = train.tax

model1 = LinearRegression(positive=False)
model1.fit(x1, y1)

model2 = LinearRegression(positive=False)
model2.fit(x2,y2)

def ensemble_predict(d2array, m1=model1, m2=model2):
    p1 = m1.predict(d2array)
    p1 = reshape_array(p1)
    p2 = m2.predict(p1)
    return p2