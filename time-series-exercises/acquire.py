import pandas as pd
import os
import requests


PATH = './data'

def get_json(url):
    response = requests.get(url)
    return response.json()


def walk_pages(domain, endpoint):
    out = []
    this = endpoint.split('/')[-1]
    response = get_json(domain+endpoint)
    if response['status'] == 'ok':
        payload = response['payload']
        out.extend(payload[this])
        next = payload['next_page']
        while next:
            response = get_json(domain+next)
            payload = response['payload']
            out.extend(payload[this])
            next = payload['next_page']
    return out


def new_zgulde_data():
    domain = 'https://python.zgulde.net'
    api = get_json(domain)['api']
    if api == '/api/v1':
        routes = get_json(domain+api)['payload']['routes']
        valid_enpoints = routes[::2]

        out = {}
        for endpoint in valid_enpoints:
            e = endpoint.split('/')[-1]
            t = pd.DataFrame(walk_pages(domain, api+endpoint))
            out[e] = t

        return out

    else:
        raise Exception(f'API version has been changed and may not work with this script.  Expected "/api/v1", instead got {api}')


def cache_dict(dict, path = PATH ):
    if not os.path.exists(path):
        os.makedirs(path)
    for k, v in dict.items():
        file = f'{path}/{k}.csv'
        v.to_csv(file)


def read_folder(path = PATH ):
    if os.path.exists(path):
        out = {}
        dir = os.listdir(path)
        for file in dir:
            name = file.split('.')[0]
            out[name] = pd.read_csv(f'{path}/{file}', index_col=0)
        return out


def get_zgulde(path = PATH):
    cached = False
    # Check if there is cached data to load
    if os.path.exists(path):
        dir = os.listdir(path)
        if len(dir) > 0:
            cached = True
    
    if cached:
        d = read_folder(path)
    else:
        d = new_zgulde_data()
        cache_dict(d)
    return d


def join_zgulde(dict):
    df = dict['sales'].join(
        other=dict['items'].set_index('item_id'), 
        on='item', 
        how='left').join(
            other=dict['stores'].set_index('store_id'), 
            on='store',
            how='left'
        )
    return df


def clean_zgulde(df):
    # df.sale_date = pd.to_datetime(df.sale_date, format='%a, %d %b %Y %H:%M:%S %Z')
    df.sale_date = pd.to_datetime(df.sale_date, infer_datetime_format=True)
    df = df.set_index('sale_date')
    df= df.sort_index()
    df['month'] = df.index.month
    df['day_of_week'] = df.index.day_of_week
    df['sales_total'] = df.sale_amount * df.item_price
    return df


def wrangle_zgulde(path=PATH):
    d = get_zgulde(path)
    df = join_zgulde(d)
    df = clean_zgulde(df)
    return df

def clean(varStr): 
    """Converts a string into a valid python identifier"""
    import re
    return re.sub('\W|^(?=\d)','_', varStr).lower()


def wrangle_power():
    file = 'power.csv'
    if os.path.exists(file):
        df = pd.read_csv(file, index_col=0)
    else:
        df= pd.read_csv('https://raw.githubusercontent.com/jenfly/opsd/master/opsd_germany_daily.csv')
        df.to_csv(file)

    df.columns = [clean(col) for col in df]
    df.date = pd.to_datetime(df.date)
    df = df.set_index('date')
    df = df.fillna(0)

    return df