import pandas as pd

import sklearn as sk
import sklearn.model_selection as skm
from sklearn.impute import SimpleImputer

import acquire


# from sklearn.model_selection import train_test_split
# from sklearn.impute import SimpleImputer

def prep_iris(df):
    df = df.drop(columns=['species_id','measurement_id'])
    df = df.rename(columns={'species_name' : 'species'})
    dummy = pd.get_dummies(df.species, drop_first= False)
    df = pd.concat([df, dummy], axis= 1)
    return df

def prep_titanic(df):
    '''
    This function will clean the data...
    '''
    df = df.drop_duplicates()
    cols_to_drop = ['deck', 'embarked', 'class', 'age']
    df = df.drop(columns=cols_to_drop)
    df['embark_town'] = df.embark_town.fillna(value='Southampton')
    dummy_df = pd.get_dummies(df[['sex', 'embark_town']], dummy_na=False, drop_first=True)
    df = pd.concat([df, dummy_df], axis=1)
    return df

def prep_telco(df):
    cols_to_drop = ['contract_type_id',
                'payment_type_id',
                'internet_service_type_id',
                'phone_service'                
                ]
    df = df.drop(columns=cols_to_drop)
    cols_to_bool = ['partner',
        'dependents',
        'online_security',
        'online_backup',
        'device_protection', 
        'tech_support', 
        'streaming_tv',
        'streaming_movies',
        'paperless_billing',
        'churn'
        ]
    df[cols_to_bool] = df[cols_to_bool]  == 'Yes'
    df.senior_citizen = df.senior_citizen == 1
    df.multiple_lines = df.multiple_lines.replace({'No': '1', 'Yes':'2+', 'No phone service': 'None'})
    df = df.rename(columns={'multiple_lines':'phone_lines'})
    dummies = pd.get_dummies(df[['gender', 'internet_service_type', 'payment_type', 'contract_type', 'phone_lines']], drop_first=False)
    df = pd.concat([df, dummies], axis=1)
    return df

def train_validate_test_split(df, seed=123, stratify=None):
    # First split off our testing data.
    train_and_validate, test = skm.train_test_split(
        df, 
        test_size=0.2, 
        random_state=seed, 
        stratify=( df[stratify] if stratify else None)
    )
    # Then split the remaining into train/validate data.
    train, validate = skm.train_test_split(
        train_and_validate,
        test_size=0.3,
        random_state=seed,
        stratify= (train_and_validate[stratify] if stratify else None)
    )
    return train, validate, test

def titanic_impute_age(df):
    titanic = acquire.get_titanic_data()
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    df = df.join(titanic.age, how='left')
    imputer.fit(df[['age']])
    df['age'] = imputer.transform(df[['age']])
    df['age'] = round(df.age, 0)
    return df

def split_x_y(df, target, features):
    x = df[features]
    y = df[target]
    return x, y