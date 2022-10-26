import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

#####################################################
#               CONFIG VARIABLES                    #
#####################################################

ENVFILE = './env.py'
CSV='./mall.csv'

SEED = 8

DB= 'mall_customers'
SQLquery ="""
SELECT *
FROM customers
"""

FEATURES = []
TARGETS = ['spending_score']
#####################################################
#               END CONFIG VARIABLES                #
#####################################################

# Easily load a google sheet (first tab only)
def read_google(url):
    csv_export_url = url.replace('/edit#gid=', '/export?format=csv&gid=')
    return pd.read_csv(csv_export_url)

# Construct a db URL without exposing credentials
def get_db_url(database):
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

