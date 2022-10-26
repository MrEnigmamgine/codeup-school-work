import numpy as np
import pandas as pd
import datetime



def filter_students(df):
    
    # remove staff by cohort name
    std = df.where(df['cohort_name'] != 'Staff').dropna()
    
    # create mask to filter amount of times accessed to 10 or below
    mask = std['user_id'].value_counts()
    std10less = std[std['user_id'].isin(mask.index[mask<11])]
    
    # create mask to filter by times accessed when students were active
    std10less['date_mask'] = std10less.index >= pd.Timestamp('2019-11-04')
    std_final = std10less.mask(std10less['date_mask'] == False).dropna()
    
    # send it
    return std_final



def filter_grads(df):
    
    # Filter graduates by acesss after end date
    df['mask'] = df.index >= df['end_date']
    grads = df.mask(df['mask'] != True).dropna()
    
    # remove codeup staff, whether or not they graduated 
    # codeups curriculum it biases the data as current employees
    no_staff = grads.where(grads['cohort_name'] != 'Staff').dropna()
    
    # seperate by program
    FSJ = no_staff.where(no_staff['program'] == 'Full Stack Java').dropna()
    FSPHP = no_staff.where(no_staff['program'] == 'Full Stack PHP').dropna()
    DS = no_staff.where(no_staff['program'] == 'Data Science').dropna()
    F_E = no_staff.where(no_staff['program'] == 'Front-End').dropna()
    
    return FSJ, FSPHP, DS, F_E