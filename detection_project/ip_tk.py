"""IP Toolkit"""

import pandas as pd
import acquire

def get_ip_info(ip=''):
    """For a single IP Address, get a dictionary describing the IP Address."""
    import requests
    host = 'http://www.geoplugin.net/json.gp?ip='
    response = requests.get(host+ip)
    return response.json()

def build_ip_info_df(addresses):
    import time
    out = {}
    for i, ip in enumerate(addresses):
        time.sleep(.52)
        out[i] = get_ip_info(ip)
    return pd.DataFrame.from_dict(out, orient='index')

def get_ips():
    """WARNING: This will take over an hour if not reading from cache.
    
    Builds the unleaned IP geolocation dataframe."""
    import os
    filename='ips.csv'
    
    if os.path.isfile(filename):
        ips = pd.read_csv(filename)

    else:
        ip_list = acquire.wrangle_data().ip.sort_values().unique().tolist()
        ips = build_ip_info_df(ip_list)
        ips.to_csv(filename, index=False)
    return ips
    
def wrangle_ips():

    ips = get_ips()

    ips.columns = [col[10:] for col in ips]
    cols_to_keep = ['request',
                'status',
                'delay',
                'city',
                'regionCode',
                'regionName',
                'countryCode',
                'countryName',
                'continentName',
                'latitude',
                'longitude',
                ]
    ips = ips[cols_to_keep]
    ips = ips.rename(columns={'request':'ip'})

    return ips

def merge_ip_info(df):
    ips = wrangle_ips()
    df = df.merge(ips, how='left', on='ip', suffixes=[None,'_ip'])
    return df

def wrangle_ip_merged():
    df = acquire.wrangle_data()
    df = df.reset_index()
    df = merge_ip_info(df)
    df = df.set_index('datetime')
    return df


# Util functions

def detect_country_geohop_events(df, window='1D', group_by='user_id'):
    """Given a time-indexed dataframe with ip geolocation information merged onto it, 
    list events where the group_by variable appears in different countries within the same window."""
    # First do the appropriate grouping.
    temp = df.groupby([group_by,pd.Grouper(freq=window)]).countryName.value_counts()
    
    # Then parse the grouping to collect the events.
    temp = temp.rename('vcount').reset_index()
    idxName = df.index.name
    what = df[group_by].name
    geoHop_events = {}
    key = 0
    for thing in temp[group_by].unique().tolist():
        this = temp[temp[group_by] == thing]
        log = this.groupby(idxName).countryName.nunique()
        events = log[log > 1].index.tolist()
        if len(events) > 0:
            for  event in events:
                countries = this[this.datetime == event].countryName.unique().tolist()
                entry = {   'when_start': event,
                            'when_end': event + pd.Timedelta(window),
                            'where':countries,
                            what: thing}
                geoHop_events[key]= entry
                key += 1
    return pd.DataFrame.from_dict(geoHop_events, orient='index')