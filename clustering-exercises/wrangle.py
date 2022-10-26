import os
import pandas as pd
import numpy as np

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

#####################################################
#               CONFIG VARIABLES                    #
#####################################################

ENVFILE = './env.py'
CSV='./data.csv'

SEED = 8

DB= 'zillow'
SQLquery ="""
SELECT 
    *
FROM
    zillow.properties_2017
        RIGHT JOIN zillow.predictions_2017 USING(parcelid)
        LEFT JOIN zillow.airconditioningtype using(airconditioningtypeid)
        LEFT JOIN zillow.architecturalstyletype using(architecturalstyletypeid)
        LEFT JOIN zillow.heatingorsystemtype using(heatingorsystemtypeid)
        LEFT JOIN zillow.propertylandusetype USING (propertylandusetypeid)
        LEFT JOIN zillow.typeconstructiontype USING (typeconstructiontypeid) 
        LEFT JOIN zillow.storytype USING (storytypeid)
WHERE
	propertylandusetypeid = 261
;
"""

FEATURES = []
TARGETS = []
#####################################################
#               END CONFIG VARIABLES                #
#####################################################


def get_db_url(database, hostname='', username='', password='', env=''):
    '''Creates a URL for a specific database and credential set to be used with pymysql.

    Can be used either with a set of credentials passed directly to the function or with an environment file containing the credentials.
    If both are provided, the environment file takes precedence.

    Returns:
    str: Full URL for use with a pymysql connection
    '''
    if env != '':
        d = {}
        file = open(env)
        for line in file:
            (key, value) = line.split('=')
            d[key] = value.replace('\n', '').replace("'",'').replace('"','')
        username = d['username']
        hostname = d['hostname']
        password = d['password']
    url = f'mysql+pymysql://{username}:{password}@{hostname}/{database}'
    return url

def new_data():
    """Downloads a copy of data from CodeUp's SQL Server"""
    url = get_db_url(DB,env=ENVFILE)
    df = pd.read_sql(SQLquery, url)
    return df

def get_data():
    """Returns an uncleaned copy of the telco data from telco.csv.
    If the file does not exist, grabs a new copy and creates the file.
    """
    filename = CSV
    
    # if file is available locally, read it
    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col=0)
    
    # if file not available locally, acquire data from SQL database
    # and write it as csv locally for future use
    else:
        # read the SQL query into a dataframe
        df = new_data()
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df  

def fix_dtypes(df):
    """convert_dtypes() chooses some slightly wonky data types that cause problems later.
    Fix the wonk by creating a new dataframe from the dataframe. """
    df = df.convert_dtypes()
    fix = pd.DataFrame(df.to_dict()) 
    return fix

def dropna_df(df):
    """Returns a dataframe free of null values where the columns have the proper dtypes"""
    df = df.dropna()
    df = fix_dtypes(df)
    return df

## Generic split data function
def train_test_validate_split(df, seed=SEED, stratify=None):
    """Splits data 60%/20%/20%"""
    # First split off our testing data.
    train, test_validate = train_test_split(
        df, 
        train_size=3/5, 
        random_state=seed, 
        stratify=( df[stratify] if stratify else None)
    )
    # Then split the remaining into train/validate data.
    test, validate = train_test_split(
        test_validate,
        train_size=1/2,
        random_state=seed,
        stratify= (test_validate[stratify] if stratify else None)
    )
    return train, test, validate

#####################################################
#              DATA SPECIFIC FUNCTIONS              #
#####################################################


def trim_bad_data_zillow(df):
    # If it's not single unit, it's not a single family home.
    df = df[~(df.unitcnt > 1)]
    # If the lot size is smaller than the finished square feet, it's probably bad data or not a single family home
    df = df[~(df.lotsizesquarefeet < df.calculatedfinishedsquarefeet)]
    # If the finished square feet is less than 500 it is likeley an apartment, or bad data
    df = df[~(df.calculatedfinishedsquarefeet < 500)]
    # If there are no bedrooms, likely a loft or bad data
    df = df[~(df.bedroomcnt < 1)]
    # Drop duplicate parcels
    df = df.drop_duplicates(subset='parcelid')
    return df

def handle_missing_values(df, drop_cols_threshold=0.75, drop_rows_threshold=0.75):
    threshold = int(round(drop_cols_threshold * len(df.index), 0))
    df = df.dropna(axis=1, thresh=threshold) # axis 1, or ‘columns’ : Drop columns which contain missing value
    threshold = int(round(drop_rows_threshold * len(df.columns), 0))
    df = df.dropna(axis=0, thresh=threshold) # axis 0, or ‘index’ : Drop rows which contain missing values.
    return df

def engineer_features_zillow(df):
    # By assuming that null values are the equivelant to false, we can save the column taxdelinquencyflag
    df.taxdelinquencyflag = df.taxdelinquencyflag == 'Y'
    # By converting taxdelinquencyyear to instead be a measure of how long the property has been tax deliquent we can save ourselves from dropping the null values.
    df['years_tax_delinquent'] = (2017 - (df.taxdelinquencyyear +2000).replace(2099, 1999)).fillna(0)
    # Because the overwhelming majority of these is values is 1, we are probably safe imputing the 1 in the missing values.
    df.unitcnt = df.unitcnt.fillna(1.0)
    # Fill heating type with Central because it is the most common heating type
    df.heatingorsystemdesc = df.heatingorsystemdesc.fillna('Central')
    # Create a new bathroom count by including three-quarter bathrooms
    df['bathroom_sum'] = (df.fullbathcnt + (df.threequarterbathnbr.fillna(0) *.75))
    # Convert some columns into more useful information
    df['age'] = 2017 - df.yearbuilt
    df.regionidzip = df.regionidzip.replace(399675, 99675)
    df.latitude = df.latitude / 1_000_000
    df.longitude = df.longitude / 1_000_000

    return df


def clean_zillow(df):
      
    df = trim_bad_data_zillow(df)
    
    df = engineer_features_zillow(df)

    # Drop the rest of the columns that have a large null percentage
    df = handle_missing_values(df)
    
    # Drop columns that duplicate data or leak predicitons from previous model
    df = df.drop(columns=[  'id',
                            'id.1',
                            'assessmentyear',
                            'bathroomcnt',
                            'finishedsquarefeet12',
                            'propertycountylandusecode',
                            'propertylandusedesc',
                            'propertylandusetypeid',
                            'rawcensustractandblock',
                            'censustractandblock',
                            'regionidcounty',
                            'taxamount'])

    # Drop the rows that have a null value and fix dtypes.
    df = dropna_df(df)


    # Change the dtypes of some columns
    cat_cols = ['parcelid',
                'fips',
                'regionidcity',
                'regionidzip',
                'yearbuilt',
                'heatingorsystemdesc']
    for col in cat_cols:
        df[col] = df[col].astype('category')
    df.transactiondate = pd.to_datetime(df.transactiondate)

    return df


def drop_upper_and_lower_outliers(df, cols, k=1.5):
    #function to detect and handle oulier using IQR rule
    for col in df[cols]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        upper_bound =  q3 + k * iqr
        lower_bound =  q1 - k * iqr     
        df = df[(df[col] < upper_bound) & (df[col] > lower_bound)]
    return df

def trim_outliers_zillow(df):
    outlier_cols =['taxvaluedollarcnt','lotsizesquarefeet','structuretaxvaluedollarcnt']
    df = drop_upper_and_lower_outliers(df, outlier_cols, k=1.5)
    return df

def rename_zillow(df):
    df = df.rename(columns={
        'calculatedfinishedsquarefeet' : 'structure_sqft',
        'calculatedbathnbr' : 'calc_bath',
        'lotsizesquarefeet' : 'lot_sqft',
        'structuretaxvaluedollarcnt': 'tax_structure',
        'taxvaluedollarcnt': 'tax',
        'landtaxvaluedollarcnt': 'tax_land'
        })
    return df

def zillow_impute(df):
    from sklearn.impute import KNNImputer
    kimputer = KNNImputer()
    x = df[['latitude','longitude','regionidcity']]
    y = kimputer.fit_transform(x)
    df[['latitude','longitude','regionidcity']] = y
    return df

def wrangle_zillow():
    """Living function that will change to always include the latest steps"""
    df = get_data()
    df = zillow_impute(df)
    df = clean_zillow(df)
    train, test, validate = train_test_validate_split(df)
    train = trim_outliers_zillow(train)
    train = rename_zillow(train)
    test = rename_zillow(test)
    validate = rename_zillow(validate)
    return train, test, validate


from sklearn.preprocessing import MinMaxScaler
def min_max_scale_df(df, cols=None):
    """General use function to scale a dataframe and keep it in dataframe format."""
    if cols == None:
        cols = df.columns.tolist()
    df = df[cols]
    
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns )
    
    return X