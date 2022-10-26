import os

# DS Modules
import numpy as np
import pandas as pd

SEED = 8
CSV='./data.csv'
DB= 'curriculum_logs'
SQLQUERY ="""
SELECT 
    date,
    time,
    ip,
    path as endpoint,
    user_id,
    cohort_id,
    name as cohort_name,
    slack,
    start_date,
    end_date,
    program_id
FROM
    curriculum_logs.logs
left join
	curriculum_logs.cohorts on cohort_id = id
;
"""


def get_db_url(database):
    """Formats a SQL url by using the env.py file to store credentials."""
    from env import host, user, password
    url = f'mysql+pymysql://{user}:{password}@{host}/{database}'
    return url

def new_data():
    """Downloads a copy of data from a SQL Server.
    Relies on an env.py file and the configuration of the DB and SQLQUERY variables."""
    url = get_db_url(DB)
    df = pd.read_sql(SQLQUERY, url)
    return df

def get_data():
    """Returns an uncleaned copy of the data from the CSV file defined in config.
    If the file does not exist, grabs a new copy and creates the file.
    Assumes the use of a SQL query.
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


####################
# Begin Data specific transformations
####################

def get_lesson(string):
    arr = string.split('/')
    if len(arr) > 1:
        return arr[1]
    else:
        return None

def get_module(string):
    arr = string.split('/')
    return arr[0]

def parse_module_lesson(df):
    df['module'] = df.endpoint.apply(get_module)
    df['lesson'] = df.endpoint.apply(get_lesson)
    df = df.replace({'':None})
    return df


def wrangle_data():
    df = get_data()
    df = df[df.endpoint.notna()]

    df.date = pd.to_datetime(df.date)
    df.start_date = pd.to_datetime(df.start_date)
    df.end_date = pd.to_datetime(df.end_date)
    df.time = pd.to_timedelta(df.time)
    df['datetime'] = df.date + df.time
    df = df.drop(columns=['date','time'])
    df = df.set_index('datetime')
    df = df.sort_index()
    
    pnames = {  1: 'Full Stack PHP',
            2: 'Full Stack Java',
            3: 'Data Science',
            4: 'Front-End'
            }
    df['program'] = df.program_id.replace(pnames)

    df = parse_module_lesson(df)

    return df