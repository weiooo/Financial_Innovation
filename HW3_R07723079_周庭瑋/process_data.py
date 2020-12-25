#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

# GASF
def ts2gasf(ts, max_v, min_v):
    '''
    Args:
        ts (numpy): (N, )
        max_v (int): max value for normalization
        min_v (int): min value for normalization
    Returns:
        gaf_m (numpy): (N, N)
    '''
    # Normalization : 0 ~ 1
    if max_v == min_v:
        gaf_m = np.zeros((len(ts), len(ts)))
    else:
        ts_nor = np.array((ts-min_v) / (max_v-min_v))
        # Arccos
        ts_nor_arc = np.arccos(ts_nor)
        # GAF
        gaf_m = np.zeros((len(ts_nor), len(ts_nor)))
        for r in range(len(ts_nor)):
            for c in range(len(ts_nor)):
                gaf_m[r, c] = np.cos(ts_nor_arc[r] + ts_nor_arc[c])
    return gaf_m


def get_gasf(arr):
    '''Convert time-series to gasf    
    Args:
        arr (numpy): (N, ts_n, 4)
    Returns:
        gasf (numpy): (N, ts_n, ts_n, 4)
    Todos:
        add normalization together version
    '''
    arr = arr.copy()
    gasf = np.zeros((arr.shape[0], arr.shape[1], arr.shape[1], arr.shape[2]))
    for i in range(arr.shape[0]):
        for c in range(arr.shape[2]):
            each_channel = arr[i, :, c]
            c_max = np.amax(each_channel)
            c_min = np.amin(each_channel)
            each_gasf = ts2gasf(each_channel, max_v=c_max, min_v=c_min)
            gasf[i, :, :, c] = each_gasf
    return gasf


def get_arr_ohlc(data, signal, d=None):
    if signal != 'n' and signal != 'None':
        df_es = data.loc[data[signal]==1]
    elif signal == 'None':
        df_es = data.loc[data[signal]==1][:10000]
    else:
        df_es = d
    arr = np.zeros((df_es.shape[0], 10, 4))
    for index, N in zip(df_es.index, range(df_es.shape[0])):
        df = data.loc[data.index <= index][-10::]
        arr[N, :, 0] = df['open']
        arr[N, :, 1] = df['high']
        arr[N, :, 2] = df['low']
        arr[N, :, 3] = df['close']
    return arr

def get_arr_curl(data, signal, d=None):
    if signal != 'n' and signal != 'None':
        df_es = data.loc[data[signal]==1]
    elif signal == 'None':
        df_es = data.loc[data[signal]==1][:10000]
    else:
        df_es = d
    arr = np.zeros((df_es.shape[0], 10, 4))
    for index, N in zip(df_es.index, range(df_es.shape[0])):
        df = data.loc[data.index <= index][-10::]
        arr[N, :, 0] = df['close']
        arr[N, :, 1] = df['ushadow_width']
        arr[N, :, 2] = df['realbody']
        arr[N, :, 3] = df['lshadow_width']
    return arr
    
    
def process(file):
    data = pd.read_csv(file)
    data['date'] = pd.to_datetime(data['date'], format="%Y-%m-%d %H:%M:%S.%f")
    data.set_index('date', inplace=True)
    return data


def detect(data, datatype, signal, d=None):
    if datatype == 'curl':
        arr = get_arr_curl(data, signal, d)
    else:
        arr = get_arr_ohlc(data, signal, d)
    gasf = get_gasf(arr)
    return gasf


def partial_data(arr0, arr1, arr2):
    N_train1, N_train2 = 4000, 2000
    N_val1, N_val2 = 800, 400
    N_test1, N_test2 = 800, 400

    train_arr = np.concatenate((arr0[:N_train1], arr1[:N_train2], arr2[:N_train2]))
    val_arr = np.concatenate((arr0[N_train1: N_train1+N_val1], 
                              arr1[N_train2: N_train2+N_val2], 
                              arr2[N_train2: N_train2+N_val2]))

    test_arr = np.concatenate((arr0[N_train1+N_val1: N_train1+N_val1+N_test1], 
                               arr1[N_train2+N_val2: N_train2+N_val2+N_test2], 
                               arr2[N_train2+N_val2: N_train2+N_val2+N_test2]))
    
    return train_arr, val_arr, test_arr

def data_csv2dict(file, datatype):
    data = process(file)
    
    signal0, signal1, signal2 = 'None', 'Hammer', 'HangingMan'
    gasf_sig0 = detect(data, datatype, signal0, d=None)
    gasf_sig1 = detect(data, datatype, signal1, d=None)
    gasf_sig2 = detect(data, datatype, signal2, d=None)

    N_none = gasf_sig0.shape[0]
    N_hammer = gasf_sig1.shape[0]
    N_hangingman = gasf_sig2.shape[0]

    # Create the label array
    label0 = np.array([[0]]*N_none, dtype = float)
    label1 = np.array([[1]]*N_hammer, dtype = float)
    label2 = np.array([[2]]*N_hangingman, dtype = float)

    # Create the label_arr array
    label_arr0 = np.array([[1, 0, 0]]*N_none, dtype = float)
    label_arr1 = np.array([[0, 1, 0]]*N_hammer, dtype = float)
    label_arr2 = np.array([[0, 0, 1]]*N_hangingman, dtype = float)
    
    train_label, val_label, test_label = partial_data(label0, label1, label2)
    train_label_arr, val_label_arr, test_label_arr = partial_data(label_arr0, label_arr1, label_arr2)
    train_gasf, val_gasf, test_gasf = partial_data(gasf_sig0, gasf_sig1, gasf_sig2)

    ALL_arr_name = ['train_gasf', 'val_gasf', 'test_gasf',
                    'train_label', 'val_label', 'test_label',
                    'train_label_arr', 'val_label_arr', 'test_label_arr']

    ALL_arr = [train_gasf, val_gasf, test_gasf,
               train_label, val_label, test_label,
               train_label_arr, val_label_arr, test_label_arr]

    data_dict = {}
    for i in range(len(ALL_arr)):
        data_dict[ALL_arr_name[i]] = ALL_arr[i]
        
    return data_dict

