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

def dropna_df(df):
    """Returns a dataframe free of null values where the columns have the proper dtypes"""
    df = df.dropna()
    df = df.convert_dtypes()
    # convert_dtypes() chooses some slightly wonky data types that cause problems later.
    # Fix the wonk by creating a new dataframe from the dataframe.
    fix = pd.DataFrame(df.to_dict()) 
    return fix

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

def clean_zillow(df):
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

    # Drop the rest of the columns that have a large null percentage
    nullPercent = df.isna().mean().sort_values()
    dropcols = nullPercent[nullPercent > .34].index.tolist()
    df = df.drop(columns=dropcols)
    
    # Drop columns that duplicate data or leak predicitons from previous model
    df = df.drop(columns=[  'id',
                            'id.1',
                            'assessmentyear',
                            'bathroomcnt',
                            'finishedsquarefeet12',
                            'logerror',
                            'propertycountylandusecode',
                            'propertylandusedesc',
                            'propertylandusetypeid',
                            'rawcensustractandblock',
                            'censustractandblock',
                            'regionidcounty',
                            'transactiondate',
                            'taxamount'])

    # Drop the rows that have a null value.
    df = dropna_df(df)

    # Convert some columns into more useful information
    df['age'] = 2017 - df.yearbuilt
    df.regionidzip = df.regionidzip.replace(399675, 99675)
    df.latitude = df.latitude / 1_000_000
    df.longitude = df.longitude / 1_000_000

    # Change the dtypes of some columns
    cat_cols = ['parcelid',
                'fips',
                'regionidcity',
                'regionidzip',
                'yearbuilt',
                'heatingorsystemdesc']
    for col in cat_cols:
        df[col] = df[col].astype('category')

    # Drop duplicate parcels
    df = df.drop_duplicates(subset='parcelid')

    return df



def trim_zillow(df):
    # Trim the tax value outliers
    uplim = np.percentile(df.taxvaluedollarcnt, 99)
    uplim_mask = df.taxvaluedollarcnt > uplim
    df = df[~uplim_mask]
    # Trim the lot squarefoot outliers
    uplim = np.percentile(df.lotsizesquarefeet, 99)
    uplim_mask = df.lotsizesquarefeet > uplim
    df = df[~uplim_mask]
    # Trim the structure squarefoot outliers
    uplim = np.percentile(df.calculatedfinishedsquarefeet, 99)
    uplim_mask = df.calculatedfinishedsquarefeet > uplim
    df = df[~uplim_mask]
    # Trim some *probably* bad data
    df = df[~df.lotsizesquarefeet < df.calculatedfinishedsquarefeet ]
    df = df[df.bedroomcnt > 0]

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

def wrangle_zillow():
    """Living function that will change to always include the latest steps"""
    df = get_data()
    df = clean_zillow(df)
    train, test, validate = train_test_validate_split(df)
    train = trim_zillow(train)
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