#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2019 maxvelasques

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import sys
import os
from matplotlib import cm
from matplotlib import colorbar
from matplotlib import colors
from matplotlib import figure

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as optimize 
import lasio
import scipy.signal as signal

try:
    import tensorflow.python as tf
    from tensorflow.keras.losses import MAPE

    from tensorflow.python.keras import backend as K
    from tensorflow.python.keras.models import load_model
    # from tensorflow.python.framework.ops import disable_eager_execution
except:
    print('Tensorflow not available. Install tensorflow 2.0 before generating logs')

# i f 'A' in df.columns:
##############################################################################
# READ LAS
##############################################################################
DEPT_COL  = ['DEPT']
GR_COL    = ['SGRC']
RHOB_COL  = ['SBDC']
NPHI_COL  = ['FSTP']
DT_COL    = ['FDTU']
CALI_COL  = ['ACAL']
RESI_COL  = ['SEDP']

def Correct_Columns(df):
    for column in df.columns:
        if column in DEPT_COL:
            df = df.rename(columns={column:'DEPT'})
        if column in GR_COL:
            df = df.rename(columns={column:'GR'})
        if column in RHOB_COL:
            df = df.rename(columns={column:'RHOB'})
        if column in NPHI_COL:
            df = df.rename(columns={column:'NPHI'})
        if column in DT_COL:
            df = df.rename(columns={column:'DT'})
        if column in CALI_COL:
            df = df.rename(columns={column:'CALI'})
        if column in RESI_COL:
            df = df.rename(columns={column:'RESI'})
    return df

def ReadLAS(INPUTFILE):
    las = lasio.read(INPUTFILE,ignore_header_errors=True)
    df = las.df()    
    df = df.dropna()
    df = df.reset_index(drop=False)

    df = Correct_Columns(df)

    if 'DT' in df.columns:
        df = df[['DEPT','GR', 'RHOB', 'NPHI', 'DT', 'CALI', 'RESI']]
    else:
        df['DT'] = df['GR']*0.0
        df = df[['DEPT','GR', 'RHOB', 'NPHI', 'DT', 'CALI', 'RESI']]

    df = df[df['GR']   >= 0.0]
    df = df[df['RHOB'] >= 0.0]
    df = df[df['NPHI'] >= 0.0]
    df = df[df['DT']   >= 0.0]
    df = df[df['CALI'] >= 0.0]
    df = df[df['RESI'] >= 0.0]

    #CONVERT UNIT
    if os.path.split(INPUTFILE)[1] == 'MONTY_1_LWD_61_2492.LAS':
        print('   Correcting CALIPER unit from mm to inch...')
        df.loc[:,'CALI']   = df['CALI']/25.4
        
    return df

def ReadASC(INPUTFILE):
    df = pd.read_csv(INPUTFILE,delim_whitespace=True, comment='#')
    df = df.dropna()

    df = Correct_Columns(df)

    if 'DT' in df.columns:
        df = df[['DEPT','GR', 'RHOB', 'NPHI', 'DT', 'CALI', 'RESI']]
    else:
        df['DT'] = df['GR']*0.0
        df = df[['DEPT','GR', 'RHOB', 'NPHI', 'DT', 'CALI', 'RESI']]

    df = df[df['GR']   >= 0.0]
    df = df[df['RHOB'] >= 0.0]
    df = df[df['NPHI'] >= 0.0]
    df = df[df['DT']   >= 0.0]
    df = df[df['CALI'] >= 0.0]
    df = df[df['RESI'] >= 0.0]
    return df.reset_index(drop=True)

def ReadCoords(INPUTFILE):
    df = pd.read_excel(INPUTFILE)
    df = df[['Filename', 'Well', 'Latitude', 'Longitude']]
    df = df.dropna()
    return df

def WriteLAS(OUTPUTFILE,df):
    las = lasio.LASFile()
    las.other = 'LAS file created from scratch using lasio'

    for column in df.columns:
        las.add_curve(column, df[column].values,descr='fake data')
    las.write(OUTPUTFILE, version=2.0)


##############################################################################
# PLOT FUNCTIONS;
##############################################################################
def emptyplotLog(fig,axes):
    data = np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    emptylog = pd.DataFrame(columns=['DEPT', 'SGRC', 'SBDC', 'FSTP', 'FDTU', 'ACAL', 'SEDP'])
    emptylog = emptylog.set_index('DEPT',drop=False)
    emptylog = emptylog.rename(columns={"index": "DEPT"})
    emptylog = emptylog.rename(columns={"SGRC": "GR", "SBDC": "RHOB", "FSTP": "NPHI", "FDTU": "DT", "ACAL": "CALI", "SEDP": "RESI",})
    emptylog.loc[0] = data
    plotLogs('empty',fig,axes,emptylog,emptylog)

def plotLogs(filename,fig,axes ,welllog, welllog_pre = None, welllog_ffnn = None, welllog_random_forest = None, welllog_gardner = None,paramstr="",errorstr=""):
    if isinstance(welllog_pre, pd.DataFrame):
        prepross = True
    else:
        prepross = False

    if isinstance(welllog_ffnn, pd.DataFrame):
        ffnn = True
    else:
        ffnn = False

    if isinstance(welllog_random_forest, pd.DataFrame):
        random_forest = True
    else:
        random_forest = False

    if isinstance(welllog_gardner, pd.DataFrame):
        gardner = True
    else:
        gardner = False

    ax = axes[0]
    if filename == 'empty':
        ax.set_ylim([1000,0])
    else:
        ax.set_ylim([max(welllog['DEPT']),min(welllog['DEPT'])])
    ax.grid(which='both')
    if prepross:
        ax.plot(welllog['GR']    , welllog['DEPT']    , label='Original',  color='black',linewidth=1.2, linestyle='-')
        ax.plot(welllog_pre['GR'], welllog_pre['DEPT'], label='Edited'  ,  color='cyan' ,linewidth=0.8, linestyle='--')
    else:
        ax.plot(welllog['GR']    , welllog['DEPT']    , label='Original',  color='cyan',linewidth=1.2, linestyle='-')

    ax.set_xlabel('GR (API)')
    ax.set_ylabel('DEPTH (m)')
    if filename != 'empty':
        if prepross:
            ax.legend(loc='lower right')
    
    ax = axes[1]
    ax.grid(which='both')
    if prepross:
        ax.plot(welllog['RHOB']    , welllog['DEPT']    , label='Original',  color='black',linewidth=1.2, linestyle='-')
        ax.plot(welllog_pre['RHOB'], welllog_pre['DEPT'], label='Edited'  ,  color='red' ,linewidth=0.8, linestyle='--')   
    else:
        ax.plot(welllog['RHOB']    , welllog['DEPT']    , label='Original',  color='red',linewidth=1.2, linestyle='-')
    ax.set_xlabel('RHOB (g/cc)')
    if filename != 'empty':
        if prepross:
            ax.legend(loc='lower right')
            
    ax = axes[2]
    ax.grid(which='both')
    if prepross:
        ax.plot(welllog['NPHI']    , welllog['DEPT']    , label='Original',  color='black',linewidth=1.2, linestyle='-')
        ax.plot(welllog_pre['NPHI'], welllog_pre['DEPT'], label='Edited'  ,  color='yellow' ,linewidth=0.8, linestyle='--')
    else:
        ax.plot(welllog['NPHI']    , welllog['DEPT']    , label='Original',  color='yellow',linewidth=1.2, linestyle='-')
    ax.set_xlabel('NPHI (pu)')
    if filename != 'empty':
        if prepross:
            ax.legend(loc='lower right')
            
    ax = axes[3]
    ax.grid(which='both')
    ax.plot(welllog['CALI']    , welllog['DEPT']    , label='Original',  color='magenta',linewidth=1.2, linestyle='-')
    ax.set_xlabel('CALI (in)')
    if filename != 'empty':
        if prepross:
            ax.legend(loc='lower right')
            
    ax = axes[4]
    ax.grid(which='both')
    if prepross:
        ax.plot(welllog['RESI']    , welllog['DEPT']    , label='Original',  color='black',linewidth=1.2, linestyle='-')
        ax.plot(welllog_pre['RESI'], welllog_pre['DEPT'], label='Edited'  ,  color='orange' ,linewidth=0.8, linestyle='--')
    else:
        ax.plot(welllog['RESI']    , welllog['DEPT']    , label='Original',  color='orange',linewidth=1.2, linestyle='-')
    ax.set_xlabel('RESI (ohm.m)')
    if filename != 'empty':
        if prepross:
            ax.legend(loc='lower right')

    ax = axes[5]
    ax.grid(which='both')
    
    if ffnn or gardner or random_forest:
        if prepross == False:   
            ax.plot(welllog['DT']    , welllog['DEPT']    , label='Original',  color='black',linewidth=1.2, linestyle='-')
    else:
        ax.plot(welllog['DT']    , welllog['DEPT']    , label='Original',  color='black',linewidth=1.2, linestyle='-')

    if prepross:
        if ffnn or gardner:
            ax.plot(welllog_pre['DT'],     welllog_pre['DEPT'],     label='Edited'   ,  color='black'   ,linewidth=1.2, linestyle='-')
        else:
            ax.plot(welllog_pre['DT'],     welllog_pre['DEPT'],     label='Edited'   ,  color='magenta' ,linewidth=0.8, linestyle='--')

    if gardner:
        ax.plot(welllog_gardner['DT'], welllog_gardner['DEPT'], label='Gardner'  ,  color='blue'  ,linewidth=0.8, linestyle=':')

    if ffnn:
        ax.plot(welllog_ffnn['DT'],    welllog_ffnn['DEPT'],    label='FFNN'     ,  color='red'   ,linewidth=0.8, linestyle='--')

    if random_forest:
        ax.plot(welllog_random_forest['DT'],    welllog_random_forest['DEPT'],    label='Random Forest'     ,  color='cyan'   ,linewidth=0.8, linestyle='-.')

    ax.set_xlabel('DT ($\mu$s/ft)')
    if filename != 'empty':
        if prepross or ffnn or gardner or random_forest:
            ax.legend(loc='lower right')

    fig.tight_layout()
    fig.subplots_adjust(top=0.95, bottom=0.12, left=0.05, right=0.95)
    
    if ffnn or gardner or random_forest:
        fig.text(0.01,0.04,"Pre-processing parameters: " + paramstr,fontweight='bold')
        fig.text(0.01,0.01,'Mean Absolute Percentage Error for predicted log: ' + errorstr,fontweight='bold',color='red')
    else:
        fig.text(0.01,0.01,"Pre-processing parameters: " + paramstr,fontweight='bold')

def basemapplot(fig,axes,coords,welllogs):
    ax = axes[0]
    ax.set_xlabel('Longitude (˚)')
    ax.set_ylabel('Latitude (˚)')
    ax.set_zlabel('Depth (m)')
    ax.grid(which='both')
    ax.set_zlim(3000,0)

    for index, row in coords.iterrows(): 
        df = welllogs[row['Filename']]
        ax.scatter(row['Longitude'] ,row['Latitude'] , label=row['Well'], marker='x',)
        ax.plot([row['Longitude'],row['Longitude']], [row['Latitude'],row['Latitude']], [0.0,           df.DEPT.min()],color='gray', ls='--', lw=0.5 )
        ax.plot([row['Longitude'],row['Longitude']], [row['Latitude'],row['Latitude']], [df.DEPT.min(), df.DEPT.max()],color='red')
    ax.legend(loc='upper left', bbox_to_anchor=(-0.15, 0.9))

##############################################################################
# PRE-PROCESSING FUNCTION;
##############################################################################
def PrepareStringStatus(current_preprocessing_pars):
    preparamstr = ''
    if current_preprocessing_pars['cut_check']:
        preparamstr += 'Cut (0 to %d meters) | ' %(current_preprocessing_pars['cut'])
    if current_preprocessing_pars['dbscan_check']:
        preparamstr += 'DBScan (EPS = %f , Min_Neig = %d) | ' %(current_preprocessing_pars['dbscan_eps'],current_preprocessing_pars['dbscan_minneigh'])
    if current_preprocessing_pars['filter_check']:
        preparamstr += 'Median filter size (%d samples)' %(current_preprocessing_pars['filter_size'])
    return preparamstr

def PrepareStringError(ffnn_error,random_forest_error,gardner_error):
    errorstr = ''
    if gardner_error > 0:
        if gardner_error >= 100:
            errorstr = errorstr + "Gardner = Not available | "
        else:
            errorstr = errorstr + "Gardner = %.2f%% | " %(gardner_error)
    if ffnn_error > 0:
        if ffnn_error >= 100:
            errorstr = errorstr + "FFNN = Not available | "
        else:
            errorstr = errorstr + "FFNN = %.2f%% | " %(ffnn_error)
    if random_forest_error > 0:
        if random_forest_error >= 100:
            errorstr = errorstr + "Random Forest = Not available | "
        else:
            errorstr = errorstr + "Random Forest = %.2f%% | " %(random_forest_error)
    return errorstr

def concatenate_pandas(df):
    first = True
    for key in df.keys():
        if first:
            tmp = df[key].copy()
            tmp['filename'] = key
            result = tmp
            first = False
        else:
            tmp = df[key].copy()
            tmp['filename'] = key
            result = pd.concat([result, tmp],sort=True)
    return result.reset_index(drop=True)

def split_pandas(df):
    keys = pd.unique(df['filename'])
    result = {}
    for key in keys:
        tmp = df[ df['filename'] == key]
        result[key] = tmp.copy()
        result[key] = result[key].drop(columns=['filename'])
        result[key] = result[key].reset_index(drop=True) #.set_index('DEPT',drop=True)
    return result

def dicts_are_equal(dict1,dict2):
    for key in dict1.keys():
        if dict1[key] != dict2[key]:
            return False 
    return True

##############################################################################
# PREDICT FUNCTIONS
##############################################################################

# def mean_absolute_percentage_error(y_true, y_pred):
#     if np.sum(np.absolute(y_true)) > 0.0:
#         return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#     else:
#         return 100


def ffnn_predict(queue,welllog):
    features_columns = ['GR','RHOB','NPHI','RESI']
    target_column   = 'DT'
    try:
        from joblib import dump, load
    except:
        print('Joblib not available. Install joblib before generating logs')

    try:
        from sklearn.preprocessing import MinMaxScaler
    except:
        print('Sklearn not available. Install sklearn before generating logs')

    ffnn_model = load_model('models/ffnn_model.h5')
    ffnn_X_scaler = load('models/ffnn_model.X_scaler')
    ffnn_y_scaler = load('models/ffnn_model.y_scaler')
    ffnn_model.summary()

    X_test_norm = ffnn_X_scaler.transform(welllog[features_columns].values)
    y_predict_norm = ffnn_model.predict(X_test_norm)
    y_predict = ffnn_y_scaler.inverse_transform(y_predict_norm).flatten()

    welllog_dept = pd.DataFrame(data=welllog['DEPT'].values, columns=['DEPT'])
    welllog_ffnn = pd.DataFrame(data=y_predict, columns=[target_column])
    welllog_ffnn = pd.concat([welllog_dept,welllog_ffnn],axis=1)

    K.clear_session()
    y_true = welllog[target_column].values
    # y_true_norm = ffnn_y_scaler.transform(y_true.reshape(1,-1))

    result = {}
    result['log'] = welllog_ffnn
    result['error'] = MAPE(y_true, y_predict)
    queue.put(result)

    # return welllog_ffnn, mean_absolute_percentage_error(y_true, y_predict)


def gardner_predict(welllog):
    target_column   = 'DT'
    amp   = 570.8622481095668
    index = -2.0819097333683785
    y_predict = amp*(welllog['RHOB'].values**index)
    
    welllog_dept = pd.DataFrame(data=welllog['DEPT'].values, columns=['DEPT'])
    welllog_gardner = pd.DataFrame(data=y_predict, columns=[target_column])
    welllog_gardner = pd.concat([welllog_dept,welllog_gardner],axis=1)

    y_true = welllog[target_column].values
    return welllog_gardner, MAPE(y_true, y_predict)


def random_forest_predict(welllog):
    features_columns = ['RHOB','RESI','GR','NPHI']
    target_column   = 'DT'
    try:
        import pickle
    except:
        print('Pickle not available. Install pickle before generating logs with Random Forest model')

    rfr_X_scaler = pickle.load(open('models/rfr_model.X_scaler', 'rb'))
    rfr_y_scaler = pickle.load(open('models/rfr_model.y_scaler', 'rb'))
    rfr_model = pickle.load(open('models/rfr_model.h5', 'rb'))

    X_test_norm = rfr_X_scaler.transform(welllog[features_columns].values)
    y_predict_norm = rfr_model.predict(X_test_norm)
    y_predict = rfr_y_scaler.inverse_transform(y_predict_norm.reshape((-1, 1))).flatten()

    welllog_dept = pd.DataFrame(data=welllog['DEPT'].values, columns=['DEPT'])
    welllog_rfr = pd.DataFrame(data=y_predict, columns=[target_column])
    welllog_rfr = pd.concat([welllog_dept,welllog_rfr],axis=1)

    y_true = welllog[target_column].values

    return welllog_rfr, MAPE(y_true, y_predict)

# def ffnn_predict(welllog, progressdialog):
#     # keras.backend.set_learning_phase(0)
#     # tf.config.threading.set_inter_op_parallelism_threads(1)
#     # disable_eager_execution()
#     ffnn_model = load_model('models/ffnn_model.h5')
#     # ffnn_model._make_predict_function()
#     ffnn_X_scaler = load('models/ffnn_model.X_scaler')
#     ffnn_y_scaler = load('models/ffnn_model.y_scaler')
#     ffnn_model.summary()

#     # ffnn_model.predict(np.array([[0,0,0,0,0,0]])) # warmup
#     # session = K.get_session()
#     # graph = tf.get_default_graph()
#     # graph = K.get_graph()

#     progressdialog.Update(50)

#     X_test_norm = ffnn_X_scaler.transform(welllog[features_columns].values)
#     # with session.as_default():
#     #     with graph.as_default():
#     y_predict_norm = ffnn_model.predict(X_test_norm)
#     # graph.finalize() # finalize

#     y_predict = ffnn_y_scaler.inverse_transform(y_predict_norm)

#     progressdialog.Update(70)

#     welllog_ffnn = pd.DataFrame(data=y_predict, columns=target_column)
#     welllog_ffnn = pd.concat([welllog['DEPT'],welllog_ffnn],axis=1)

#     K.clear_session()

#     # ffnn_model = None
#     # ffnn_X_scaler = None
#     # ffnn_y_scaler = None
#     # import gc
#     # gc.collect()
#     return welllog_ffnn