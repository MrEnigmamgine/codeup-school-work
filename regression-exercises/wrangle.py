import os
import pandas as pd

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

#####################################################
#               CONFIG VARIABLES                    #
#####################################################

ENVFILE = './env'
CSV='./data.csv'

SEED = 8

DB= 'zillow'
SQLquery ="""
SELECT 
    id,
    bedroomcnt,
    bathroomcnt,
    calculatedfinishedsquarefeet,
    taxvaluedollarcnt,
    yearbuilt,
    taxamount,
    fips
FROM
    zillow.properties_2017
        JOIN
    propertylandusetype USING (propertylandusetypeid)
WHERE
    propertylandusetypeid = 261
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

def get_data_dropna():
    """Returns a dataframe free of null values where the columns have the proper dtypes"""
    df = get_data()
    df = df.dropna()
    df = df.convert_dtypes()
    # convert_dtypes() chooses some slightly wonky data types that cause problems later.
    # Fix the wonk by creating a new dataframe from the dataframe.
    fix = pd.DataFrame(df.to_dict()) 
    return fix

## Generic split data function
def train_validate_test_split(df, seed=SEED, stratify=None):
    """Splits data 60%/20%/20%"""
    # First split off our testing data.
    train, test_validate = train_test_split(
        df, 
        test_size=3/5, 
        random_state=seed, 
        stratify=( df[stratify] if stratify else None)
    )
    # Then split the remaining into train/validate data.
    test, validate = train_test_split(
        test_validate,
        test_size=1/2,
        random_state=seed,
        stratify= (test_validate[stratify] if stratify else None)
    )
    return train, test, validate



def wrangle_zillow():
    """Living function that will change to always include the latest steps"""
    df = get_data_dropna()
    df['fips'] = df['fips'].astype('object')
    df = df.drop(columns=['id','taxamount'])
    df['age'] = 2022 - df.yearbuilt
    # df = df.fillna(0)
    train, validate, test = train_validate_test_split(df)
    return train, validate, test


def scale_zillow(train, validate, test):
    """Takes 3 (zillow) dataframes, trains a quantile scaler on the first, then transforms them all to fit a normal distribution."""
    # Define the columns to be scaled
    scalecols = [   'bedroomcnt',
                    'bathroomcnt',
                    'calculatedfinishedsquarefeet',
                    'taxvaluedollarcnt',
                    'age'
                ]
    # Create and fit the model to the train sample set
    
    scaler = QuantileTransformer(output_distribution='normal')
    scaler.fit(train[scalecols])
    # Transform each sample set
    train[scalecols] = scaler.transform(train[scalecols])
    test[scalecols] = scaler.transform(test[scalecols])
    validate[scalecols] = scaler.transform(validate[scalecols])
    # Return
    return train, validate, test

def wrangle_scale_zillow():
    """Fun"""
    train, validate, test = wrangle_zillow()
    train, validate, test = scale_zillow(train, validate, test)
    return train, validate, test

from sklearn.preprocessing import MinMaxScaler
def min_max_scale_df(df, cols=None):
    """General use function to scale a dataframe and keep it in dataframe format."""
    if cols == None:
        cols = df.columns.tolist()
    df = df[cols]
    
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns )
    
    return X