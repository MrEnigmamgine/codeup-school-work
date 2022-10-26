import os
import pandas as pd
import numpy as np

import mytk # My Toolkit

from sklearn.model_selection import train_test_split

#####################################################
#               CONFIG VARIABLES                    #
#####################################################

ENVFILE = './env.py'
CSV='./data.csv'

SEED = 8

DB= 'superstore_db'
SQLquery ="""
SELECT 
    *
FROM
    orders as ord
	join
		regions using (`Region ID`)
	join
		customers using (`Customer ID`)
	join 
		categories using (`Category ID`)
	join
		products using (`Product ID`)
;
"""

FEATURES = []
TARGETS = []
#####################################################
#               END CONFIG VARIABLES                #
#####################################################

# Easily load a google sheet (first tab only)
def read_google(url):
    csv_export_url = url.replace('/edit#gid=', '/export?format=csv&gid=')
    return pd.read_csv(csv_export_url)

# Construct a db URL without exposing credentials
def get_db_url(database):
    """Formats a SQL url by using the env.py file to store credentials."""
    from env import host, user, password
    url = f'mysql+pymysql://{user}:{password}@{host}/{database}'
    return url


def new_data():
    """Downloads a copy of data from CodeUp's SQL Server"""
    url = get_db_url(DB)
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


#####################################################
#              DATA SPECIFIC FUNCTIONS              #
#####################################################

def wrangle_superstore():
    df = get_data()
    # Rename columns for ease of typing
    df.columns = [mytk.clean(col) for col in df]
    # Drop id columns and country because useless
    df = df.drop(columns=['category_id', 'region_id', 'country'])
    # Convert time columns to datetime datatypes
    df.order_date = pd.to_datetime(df.order_date)
    df.ship_date = pd.to_datetime(df.ship_date)

    return df