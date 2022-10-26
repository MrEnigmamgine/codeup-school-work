import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor

import wrangle
# import my_toolkit

SEED = 8

class BaselineRegressor:
    """ A simple class meant to mimic sklearn's modeling methods so that I can standardize my workflow.
    Assumes that you are fitting a single predictor.  
    For multiple predictors you will need multiple instances of this class.
    
    TODO: Handle multi-dimensional predictors
    TODO: Handle saving feature names
    """
    def __init__(self):
        """This isn't needed, but I'm leaving this here to remind myself that it's a thing."""
        pass


    def fit(self, y):
        """Calculates the mean for the target variable and assigns it to this instance."""
        if len(y.shape) == 1:
            self.baseline = y.mean()
        else:
             raise ValueError('Expected a 1 dimensional array.')

    def predict(self, x):
        """Always predicts the mean value."""
        n_predictions = len(x)
        return np.full((n_predictions), self.baseline)

def regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict:

    
    y = actual
    yhat = predicted
    resid_p = y - yhat
    sum_of_squared_errors = (resid_p**2).sum()

    error_metrics = {
        'max_error': metrics.max_error(actual, predicted),
        'sum_squared_error' : sum_of_squared_errors,
        'mean_squared_error' : metrics.mean_squared_error(actual, predicted),
        'root_mean_squared_error' : metrics.mean_squared_error(actual, predicted, squared=False),
        'mean_aboslute_error' : metrics.mean_absolute_error(actual, predicted),
        'r2_score' : metrics.r2_score(actual, predicted, force_finite=False)
    }

    return error_metrics

def build_kmeans_clusterer(df, cols, k, seed=SEED):
    from sklearn.cluster import KMeans
    clusterer = KMeans(n_clusters=k, random_state=seed)
    clusterer.fit(df[cols])
    return clusterer


# def get_kmeans_clusters(df, cols, k, clusterer=None):
#     if clusterer == None:
#         from sklearn.cluster import KMeans
#         clusterer = KMeans(n_clusters=k)
#         clusterer.fit(df[cols])
#     s = clusterer.predict(df[cols])
#     return s


def make_minmax_scaler(df, cols):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(df[cols])
    return scaler


def df_scale_cols(df, cols, scaler, reverse=False):
    out = df.copy()
    if reverse:
        out[cols] = scaler.inverse_transform(out[cols])
    else:
        out[cols] = scaler.transform(out[cols])
    return out





## Import the data
train, test, validate = wrangle.wrangle_zillow()
## Make a backup in case I mess something up later
orig_samples = {
    'train': train.copy(),
    'test': test.copy(),
    'validate': validate.copy()
}

scale_cols = [
 'bedroomcnt',
 'calc_bath',
 'structure_sqft',
 'fullbathcnt',
 'latitude',
 'longitude',
 'lot_sqft',
 'roomcnt',
 'tax_structure',
 'tax',
 'tax_land',
 'years_tax_delinquent',
 'bathroom_sum',
 'age',
 ]


scaler1 = make_minmax_scaler(train, scale_cols)

train = df_scale_cols(train, scale_cols, scaler1)
test = df_scale_cols(test, scale_cols, scaler1)
validate = df_scale_cols(validate, scale_cols, scaler1)

cluster1_cols = ['age','tax']
clusterer1 = build_kmeans_clusterer(train, cluster1_cols, k=6)

cluster2_cols = ['latitude','longitude','age', 'tax']
clusterer2 = build_kmeans_clusterer(train, cluster2_cols, k=6)

cluster3_cols = ['latitude','longitude','structure_sqft','lot_sqft']
clusterer3 = build_kmeans_clusterer(train, cluster3_cols, k=5)

all_clusterers = [(clusterer1, cluster1_cols), (clusterer2, cluster2_cols), (clusterer3, cluster3_cols)]

def df_add_clusters(df: pd.DataFrame, clusterer_col_tuples: list):
    
    out = df.copy()
    for i, (clusterer, cols) in enumerate(clusterer_col_tuples):
        out['cluster'+str(i+1)] = clusterer.predict(out[cols])
    return out

train = df_add_clusters(train, all_clusterers)
test = df_add_clusters(test, all_clusterers)
validate = df_add_clusters(validate, all_clusterers)

ytrain = train.logerror
ytest = test.logerror
yval = validate.logerror

baseline = BaselineRegressor()
baseline.fit(ytrain)

models = {}
models['baseline'] = baseline


train_metrics = {}
test_metrics = {}
validate_metrics = {}



train_metrics['train_baseline'] = regression_metrics(ytrain, baseline.predict(ytrain))
test_metrics['test_baseline'] = regression_metrics(ytest, baseline.predict(ytest))
validate_metrics['validate_baseline'] = regression_metrics(yval, baseline.predict(yval))


train_c1_dummies = pd.get_dummies(train.cluster1, prefix='cluster1')
train_c2_dummies = pd.get_dummies(train.cluster2, prefix='cluster2')
train_c3_dummies = pd.get_dummies(train.cluster3, prefix='cluster3')
x1 = pd.concat([train_c1_dummies, train_c2_dummies, train_c3_dummies], axis=1)

test_c1_dummies = pd.get_dummies(test.cluster1, prefix='cluster1')
test_c2_dummies = pd.get_dummies(test.cluster2, prefix='cluster2')
test_c3_dummies = pd.get_dummies(test.cluster3, prefix='cluster3')
t1 = pd.concat([test_c1_dummies, test_c2_dummies, test_c3_dummies], axis=1)

validate_c1_dummies = pd.get_dummies(validate.cluster1, prefix='cluster1')
validate_c2_dummies = pd.get_dummies(validate.cluster2, prefix='cluster2')
validate_c3_dummies = pd.get_dummies(validate.cluster3, prefix='cluster3')
v1 = pd.concat([validate_c1_dummies, validate_c2_dummies, validate_c3_dummies], axis=1)

# Collect first model
# Linear Regression using only the cluster information
model = linear_model.LinearRegression()
model.fit(x1, ytrain)
modname = 'linearregression_all_clusters'
models[modname] = model
train_metrics['train_'+modname] = regression_metrics(ytrain, model.predict(x1))
test_metrics['test_'+modname] = regression_metrics(ytest, model.predict(t1))


# Collect second model
# Lasso LARS using only the cluster information
model = linear_model.LassoLars(alpha=0.1, normalize=False)
model.fit(x1, ytrain)
modname = 'lassolars_all_clusters'
models[modname] = model
train_metrics['train_'+modname] = regression_metrics(ytrain, model.predict(x1))
test_metrics['test_'+modname] = regression_metrics(ytest, model.predict(t1))



depth_range = range(1,15)

# Collect the third model sets
# Random Forest using only cluster information

for i in depth_range:
    model = RandomForestRegressor(max_depth=i)
    model.fit(x1, ytrain)
    modname = 'randomforest_all_clusters_depth_'+str(i)
    models[modname] = model
    train_metrics['train_'+modname] = regression_metrics(ytrain, model.predict(x1))
    test_metrics['test_'+modname] = regression_metrics(ytest, model.predict(t1))

# Collect the fourth model sets
# Random Forest using cluster 2 along with building and lot size

x2 = train[['structure_sqft','lot_sqft']]
x2 = pd.concat((x2, train_c2_dummies), axis=1)
t2 = test[['structure_sqft','lot_sqft']]
t2 = pd.concat((t2, test_c2_dummies), axis=1)

for i in depth_range:
    model = RandomForestRegressor(max_depth=i)
    model.fit(x2, ytrain)
    modname = 'randomforest_size_cluster2_depth_'+str(i)
    models[modname] = model
    train_metrics['train_'+modname] = regression_metrics(ytrain, model.predict(x2))
    test_metrics['test_'+modname] = regression_metrics(ytest, model.predict(t2))


# I need to evaluate something against validate for 2 points on my grade so..
modname = 'linearregression_all_clusters'
validate_metrics[modname] = regression_metrics(yval, models[modname].predict(v1))


# Wrap my collected information into dataframes for easier viewing.
df_train_metrics = pd.DataFrame.from_dict(train_metrics, orient='index')
df_test_metrics = pd.DataFrame.from_dict(test_metrics, orient='index')
df_validate_metrics = pd.DataFrame.from_dict(validate_metrics, orient='index')