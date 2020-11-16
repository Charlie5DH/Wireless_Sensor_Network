#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import datetime as dt
import matplotlib.dates as mdates
from time import time

from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from scipy import stats
import scipy.io

import pywt
from scipy.fftpack import fft
from collections import Counter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from scipy.signal import welch

# for anomalies
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Helper functions for later

def add_date_features(df):
    '''
    Function for adding date features to any dataset.
    The function first resets the index considering the Timestamp
    is set as index.
    Adds hour, day, month, week, weekday and daylight features.
    Considers day from 7:00 am to 7:00 Pm.
    '''
    # df.reset_index(inplace=True)
    # take date features from timeseries
    df['hour'] = df['Timestamp'].dt.hour
    df['day'] = df['Timestamp'].dt.day
    df['month'] = df['Timestamp'].dt.month
    df['week'] = df['Timestamp'].dt.week
    df['weekday'] = df['Timestamp'].dt.weekday
    df['daylight'] = ((df['hour'] >= 7) & (df['hour'] <= 19)).astype(int)
    df.set_index('Timestamp', drop=True, inplace=True)
    return df

def drop_power(df, Powers, value=-9):
    '''
    for dropping the -9 dbm value
    Powers is a dictionary with the names
    of the columns. Considers the original dataset.
    '''
    for ii in Powers:
        indexx = df[df[ii] == -9].index
        df = df.drop(df.index[indexx], axis=0)
        df.reset_index(inplace=True, drop=True)
    return df

def get_radios(df):
    '''
    Get the names of the radio modules in the base dataframe.
    return a list of string.
    '''
    Radios = df.Module.unique()[:7]
    Radios = np.append(Radios, df.Module.unique()[-1])
    return Radios

def get_receivers(df):
    '''
    Get the names of the receivers, this names are different than the others
    cause they are in an hexadecimal way (0x0xxx)
    '''
    return np.array(df['Receiver'].unique())

def create_RSSI_dataframe(df_powers, df, plot=True, start_date='2018-01-01', end_date='2018-02-16', raw=True, resample_time='60Min'):
    '''
    This functions returns the entire dataset with the RSSI
    aranged by receiver.
    The original dataset is aranged by Neighbours, one Neighbour can have data from
    many different modules with different RSSI levels, this makes harder the analysis
    of timeseries. This functions arange the dataset in receivers, therefor, makes easier
    the working with timeseries.

    df_powers: Dataset with RSSI values.
    df: Dataset with othere features and modules names (this can be improved)
    plot: if plot==True then plots the RSSI data from every receiver
    start_date: start date in case you want to take just a slice
    end date: ending date for slice
    raw: if raw==True the data is plotted as raw values (only applies to plot)
    resample_time: in case raw==False this is the time for resample. Default as 1Hour.

    return: dff; The dataset aranged by receivers with the raw values.

    '''
    receivers = get_receivers(df_powers)
    radios = get_radios(df)

    dff = pd.DataFrame(data=None)
    for i in range(len(receivers)):
        subset, serie, transmitters, Tx = arange_RSSI_serie(df_powers, receiver=i, start_date=start_date, end_date=end_date, plot=False, plot_entire=False)
        serie['Receiver'] = radios[i]
        dff = pd.concat([dff, serie])

    if plot==True:
        color=['b','r','m','g','c','orange', 'y', 'grey']
        fig, axes = plt.subplots(nrows=receivers.shape[0], ncols=1, figsize=(24,28), sharey=True, sharex=True)
        fig.suptitle('RSSI data from every receiver radio', x=0.5, y=1.02, fontsize=20)
        for ii, ax in zip(radios, axes):
            if raw == True:
                dff[dff['Receiver']==ii].dropna(axis=1, how='all').plot(ax=ax)
            else:
                dff[dff['Receiver']==ii].dropna(axis=1, how='all').resample(resample_time).mean().plot(ax=ax)
            ax.tick_params(labelrotation=0)
            ax.set_yticks(np.arange(-10, -100, step=-10))
            ax.set_title('RSSI from {}'.format(ii))
            ax.legend(loc='upper right')
        plt.tight_layout()

    return dff

def arange_RSSI_serie(df_powers, receiver=0, start_date='2018-01-01', end_date='2018-02-16', sharex=False, sharey=True, figsize=(24,14), joint=True, plot=True, plot_entire=True):
    '''
    This functions returns a dataframe of RSSI data aranged by transmitters.
    Returns all the data transmitted to one single receiver.

    df_powers: Dataset with RSSI values.
    receiver = 0: The receiver selected. Default is Coordinator.
    plot: if plot==True then plots the RSSI data from every receiver
    start_date: start date in case you want to take just a slice
    end date: ending date for slice
    sharex: if sharex==True then all data shares the same x axis, else, every graph
            will have different x axis.
    joint: if joint==True then all data will be plotted in a single graph
    plot_entire: if this is True, then will plot the original Neighbour to see the difference.

    return: dff; The dataset aranged by receivers with the raw values.
    '''

    Receivers = np.array(df_powers['Receiver'].unique())
    subset = df_powers[df_powers['Receiver'] == Receivers[receiver]][start_date:end_date]
    subset = subset.dropna(axis=1, how='all')

    Tx, Ptx = [], []
    for i in range(int(subset.columns[3:].shape[0]/2)):
        Tx.append(subset.columns[3:][i*2])
        Ptx.append(subset.columns[4:][i*2])

    transmitters = []
    for ii in Tx:
        for i in range(len(subset[ii].value_counts().index)):
            transmitters.append(subset[ii].value_counts().index[i])
    transmitters = pd.Series(data=transmitters)
    transmitters.drop_duplicates(inplace=True)
    transmitters = transmitters.values
    #print(transmitters, Tx, Ptx)

    serie = pd.DataFrame(data=None, index=subset.index)
    for t in transmitters:
        dff = pd.DataFrame(data=None)
        for ii,jj in zip(Tx, Ptx):
            dff = pd.concat([dff, subset[subset[ii]==t][jj]])
        dff[t] = dff
        dff.drop(0, axis=1, inplace=True)
        serie[t] = dff[t][~dff[t].index.duplicated()]

    if plot==True:
        if plot_entire == True:
            plt.figure(figsize=(24,4))
            subset['P_Tx1(dbm)'].plot()
            plt.title('RSSI from Neighbour 1 over the entire period for Receiver {}'.format(subset['Receiver'].unique()[0]))
            plt.xticks(rotation=0);
            plt.tight_layout()

        color=['b','r','m','g','c','black', 'y', 'grey']
        if joint==False:
            fig, axes = plt.subplots(nrows=transmitters.shape[0], ncols=1, figsize=figsize, sharex=sharex, sharey=sharey)
            plt.xticks(rotation=0)
            for ii, ax in zip(range(transmitters.shape[0]), axes.flat):
                serie[transmitters[ii]].plot(ax=ax, color=color[ii], label=('Transmitter '+transmitters[ii]))
                ax.tick_params(labelrotation=0)
                ax.legend()
            plt.tight_layout()
        else:
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(24,4), sharex=True, sharey=True)
            plt.xticks(rotation=0)
            plt.title('RSSI from Transmitters to Rec {}'.format(Receivers[receiver]))
            for ii in range(transmitters.shape[0]):
                serie[transmitters[ii]].plot(ax=axes, color=color[ii], label=('Transmitter '+transmitters[ii]))
                axes.tick_params(labelrotation=0)
                axes.legend(loc='upper right')
            plt.tight_layout()

    return subset, serie, transmitters, Tx

def plot_count_transmitters(dff, subset):
    plt.figure(figsize=(12,5))
    plt.title('Amounth of Data by Receiver')
    sns.countplot(x='Receiver', data=dff, palette='Set3')
    plt.tight_layout()

    Tx, Ptx = [], []
    for i in range(int(subset.columns[3:].shape[0]/2)):
        Tx.append(subset.columns[3:][i*2])

    fig, axx = plt.subplots(3,3, figsize=(30,14))
    fig.delaxes(axx[2,1])
    fig.delaxes(axx[2,2])
    plt.suptitle('Transmitters in every Neighbour sendind data to Receiver '+ subset['Receiver'].unique()[0], x=0.5, y=1.02, fontsize=20)
    if len(Tx) < 4:
        fig.delaxes(axx[2,0]);fig.delaxes(axx[1,2])
        fig.delaxes(axx[1,1]);fig.delaxes(axx[1,0])
    if len(Tx) == 4:
        fig.delaxes(axx[2,0]);fig.delaxes(axx[1,2])
        fig.delaxes(axx[1,1])
    if len(Tx) == 5:
        fig.delaxes(axx[2,0]);fig.delaxes(axx[1,2])
    if len(Tx) == 6:
        fig.delaxes(axx[2,0])
    for ii,ax in zip(Tx, axx.flat):
        sns.countplot(x=ii, data=subset, ax=ax, palette='Set2')
    plt.tight_layout()

def plot_reciprocal_RSSI(dff):
    RECEIVERS = np.array(df_powers['Receiver'].unique())
    color=['darkblue','c','g','y','r','m','orange', 'b']
    fig, axes = plt.subplots(nrows=RECEIVERS.shape[0], ncols=1, figsize=(24,28), sharey=True, sharex=True)
    fig.suptitle('Reciprocal RSSI measurement of all Radios', x=0.5, y=1.02, fontsize=20)
    for ii, ax in zip(RECEIVERS, axes):
        recipr = dff[['Receiver', ii]]
        rec = []
        for jj, c in zip(RADIOS, color):
            if recipr[recipr['Receiver']== jj].dropna(axis=1, how='all').columns.shape[0] != 1:
                recipr[recipr['Receiver']== jj].plot(ax=ax, color=c)
                ax.tick_params(labelrotation=0)
                ax.set_title('RSSI from Transmitter{}'.format(ii))
                rec.append(jj)
        ax.legend(labels=rec, loc='upper right')
        plt.tight_layout()

def dist_transmissions(df, receiver, Receivers, ax, start_date ='2018-01-01', end_date='2018-01-10'):
    '''
    Correct way to  use it:
    for rec, ax in zip(range(len(RECEIVERS)), axes.flat):
        dist_transmissions(df_powers, rec, RECEIVERS, ax)

    Plots a Distribution of the RSSI of Transmitters to Every Receiver
    '''
    subset, serie, transmitters, Tx = arange_RSSI_serie(df, receiver, joint=True, plot=False)
    for i in serie.columns:
        sns.distplot(serie[i].dropna(), kde=False, label=i, bins=20, axlabel='Receiver '+ Receivers[receiver], ax=ax)
        ax.legend()
        ax.set_xticks(np.arange(-10, -100, step=-10))
    plt.tight_layout()

def boxplot_PowerMod_date(dff, Tx='0x0057FE05', palette='Set3', date_arr=['day','weekday','week','hour'], sharey=True):
    fig, axarr = plt.subplots(nrows=1, ncols=4, figsize=(24,4), sharey=sharey)
    fig.suptitle('Transmitter '+Tx+' over different date frames',y=1.03, fontsize=15)
    #sns.boxplot(x='month',y=Neigh, data=df, ax=axarr[0], palette='Set3')
    sns.boxplot(x=date_arr[0], y=Tx, data=dff, ax=axarr[0], palette='Set3')
    sns.boxplot(x=date_arr[1], y=Tx, data=dff, ax=axarr[1], palette='Set3')
    sns.boxplot(x=date_arr[2], y=Tx, data=dff, ax=axarr[2], palette='Set3')
    sns.boxplot(x=date_arr[3], y=Tx, data=dff, ax=axarr[3], palette='Set3')

def plot_by_date(dff, by='hour', nrows=2, ncols=4, figsize=(24,7)):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=False, sharey=True)
    fig.suptitle('Mean Values by '+ by, x=0.5, y=1.02, fontsize=18)
    for i,ax in zip(dff.columns[1:9], axes.flat):
        ax.set_title('{} by {}'.format(i, by))
        data=dff.groupby(by).mean()[[i]]
        sns.lineplot(data=data, markers=True,ax=ax, err_style='bars')
        #ax.set_xticks(np.arange(0,24,2))
    plt.tight_layout()

def append_temperatures_toRSSI(df, df_powers, Modules, start_date, end_date, resample_time='30S'):
    '''
    append the temperatures of the receiver radio
    and the temperature from transmitters.
    '''
    subset, serie, transmitters, Tx = arange_RSSI_serie(df_powers, receiver=0,start_date=start_date, end_date=end_date, joint=True, plot=False)

    # get receiver name
    for i in Modules:
        if i[-2:] == subset.Receiver.unique()[0][-2:]:
            receiver = i

    # get Receiver temperature over entire period
    serie = serie.resample(resample_time).mean()
    resampled_df = df[df['Module']== receiver]['2018-01-01':'2018-01-18'].resample(resample_time).mean()[['Temp_Mod', 'VBus']]
    resampled_df = resampled_df.fillna(resampled_df.bfill())
    serie['Receiver'] = receiver
    serie[['Temp_Rece', 'VBus_Rec']] = resampled_df

    # need to do this because of the difference in the names
    mod = []
    for ii in serie.columns:
        last=ii[-2:]
        for i in Modules:
            if i[-2:] == last:
                mod.append(i)

    # transmitters temperatures
    for tx in mod:
        resampled_df = df[df['Module']== tx]['2018-01-01':'2018-01-18'].resample(resample_time).mean()[['Temp_Mod']]
        resampled_df = resampled_df.fillna(resampled_df.bfill())
        serie[['Temp_' + tx]] = resampled_df

    return serie
