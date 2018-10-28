import pandas as pd
import numpy as np
from scipy import stats
import math


def main():
    df = pd.read_csv('KS.csv')
    print(df.shape)
    df = df[df.state != 'canceled']
    df = df[df.state != 'undefined']
    df = df[df.state != 'suspended']
    df = df[df.state != 'live']
    df['launched'] = pd.to_datetime(df['launched'])
    df['deadline'] = pd.to_datetime(df['deadline'])
    df['duration_days'] = df['deadline'].subtract(df['launched'])
    df['duration_days'] = df['duration_days'].astype('timedelta64[D]')
    df = df.drop('launched', 1)
    df = df.drop('deadline', 1)
    df = df.drop('ID', 1)
    df = df.drop('name', 1)
    df = df.drop('pledged', 1)
    df = df.drop('backers', 1)
    df = df.drop('usd pledged', 1)
    df = df.drop('usd_pledged_real', 1)
    df = df.drop('usd_goal_real', 1)
    df['state'] = df['state'].map({
        'failed': 0,
        'successful': 1
        })
    df = df[df.goal > 0]
    #print(df.head)
    code = 1
    for i in df.category.unique():
        df.category.replace(i, code, inplace=True)
        code += 1

    code = 1
    for i in df.main_category.unique():
        df.main_category.replace(i, code, inplace=True)
        code += 1
    
    code = 1
    for i in df.currency.unique():
        df.currency.replace(i, code, inplace=True)
        code += 1
    
    code = 1
    for i in df.country.unique():
        df.country.replace(i, code, inplace=True)
        code += 1

    #print(df.head)    
    pre_process_manual(df)
    pre_process_grubbs(df)
    pre_process_original(df)
    
def pre_process_manual(df):
    df = df[(df['goal'] <= 100000) | ((df['goal'] >= 100000) & (df['state'] == 1)) ].copy()
    print('manual', df.shape)
    df.to_csv('KS_manual_pre_process.csv', sep=',', index=False)


def grubbs_test(N, df, a = 0.05): 
    p = 1-(a/(2*N))
    nn = N-2
    value = stats.t.ppf(p, nn)
    value**=2
    thresh = math.sqrt(value/(nn*value)) * (N-1)/math.sqrt(N)
    y = df['goal']
    mean = y.mean()
    std = y.std()
    term_factor_max = 0
    term_factor_min = 0
    mean_dev = abs(y-mean)
    for i in range(N):
        y_max = mean_dev.idxmax()
        y_min = mean_dev.idxmin()
        if y_max == term_factor_max and y_min == term_factor_min:
            break
        term_factor_max = y_max
        term_factor_min = y_min
        G1 = abs(mean_dev[y_min])/std
        G2 = abs(mean_dev[y_max])/std
        if G1>thresh:
            mean_dev = mean_dev[mean_dev != mean_dev[y_min]]
            df = df[y != y[y_min]]
        if G2>thresh:
            mean_dev = mean_dev[mean_dev != mean_dev[y_max]]
            df = df[y != y[y_max]]
        #print(i, thresh, G1, y_min, G2, y_max)
    #p = stats.t.cdf(value, nn)
    return df


def pre_process_original(df):
    df.to_csv('KS_original_pre_process.csv', sep=',', index=False)


def pre_process_grubbs(df):
    df = grubbs_test(df.shape[0], df)
    df.to_csv('KS_grubb_pre_process.csv', sep=',', index=False)


main()
