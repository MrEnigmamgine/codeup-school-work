import pandas as pd

def get_db_url(database, username='', password='', hostname='', env=''):
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

def read_google(url):
    csv_export_url = url.replace('/edit#gid=', '/export?format=csv&gid=')
    return pd.read_csv(csv_export_url)

def new_titanic_data():
    query = 'SELECT * FROM titanic_db.passengers'
    url = get_db_url('titanic_db',env='./env.py')
    df = pd.read_sql(query, url)
    return df

import os

def get_titanic_data():
    filename = "titanic.csv"
    
    # if file is available locally, read it
    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col=0)
    
    # if file not available locally, acquire data from SQL database
    # and write it as csv locally for future use
    else:
        # read the SQL query into a dataframe
        df = new_titanic_data()
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df  


def new_iris_data():
    query = """
        SELECT 
            *
        FROM
            iris_db.measurements
        JOIN
            iris_db.species using(species_id)
        ;
        """

    url = get_db_url('iris_db',env='./env.py')
    df = pd.read_sql(query, url)
    return df

import os

def get_iris_data():
    filename = "iris.csv"
    
    # if file is available locally, read it
    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col=0)
    
    # if file not available locally, acquire data from SQL database
    # and write it as csv locally for future use
    else:
        # read the SQL query into a dataframe
        df = new_iris_data()
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df  

def new_telco_data():
    query = """
        SELECT 
            *
        FROM
            telco_churn.customers
                JOIN
            telco_churn.internet_service_types USING (internet_service_type_id)
                JOIN
            telco_churn.payment_types USING (payment_type_id)
                JOIN
            telco_churn.contract_types USING (contract_type_id)
        ;
        """

    url = get_db_url('telco_churn',env='./env.py')
    df = pd.read_sql(query, url)
    return df

import os

def get_telco_data():
    filename = "telco.csv"
    
    # if file is available locally, read it
    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col=0)
    
    # if file not available locally, acquire data from SQL database
    # and write it as csv locally for future use
    else:
        # read the SQL query into a dataframe
        df = new_telco_data()
        
        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df  