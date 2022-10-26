import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

def plot_residuals(actual, predicted):
    ybar = actual.mean()
    yhat = predicted
    resid_p = actual - yhat
    resid_b = actual - ybar

    fig, ax1 = plt.subplots(1, 1, constrained_layout=True, sharey=True, figsize=(7,4))
    ax1.set_title('Predicted Residuals')
    ax1.set_ylabel('Error')
    ax1.set_xlabel('Predicted Value')
    ax1.ticklabel_format(useOffset=False, style='plain')
    ax1.scatter(x=yhat, y=resid_p)
    fig.show()



def regression_errors(actual: pd.Series, predicted: pd.Series) -> dict:
    y = actual
    ybar = actual.mean()
    yhat = predicted
    resid_p = y - yhat
    resid_b = y - ybar
    
    sum_of_squared_errors = (resid_p**2).sum()
    explained_sum_of_squares = ((yhat-ybar)**2).sum()
    total_sum_of_sqares = sum_of_squared_errors + explained_sum_of_squares
    mean_squared_error = sum_of_squared_errors / len(y)
    root_mean_squared_error = math.sqrt(mean_squared_error)
    r2_variance = (explained_sum_of_squares / total_sum_of_sqares)

    error_metrics = {
        'SSE' : sum_of_squared_errors,
        'ESS' : explained_sum_of_squares,
        'TSS' : total_sum_of_sqares,
        'MSE' : mean_squared_error,
        'RMSE' : root_mean_squared_error,
        'r2' : r2_variance
    }

    return error_metrics

def baseline_mean_errors(actual):
    baseline = np.full_like(actual, actual.mean())
    return regression_errors(actual, baseline)

def better_than_baseline(actual, predicted):
    p = regression_errors(actual, predicted)['RMSE']
    b = baseline_mean_errors(actual)['RMSE']
    return (p < b)
