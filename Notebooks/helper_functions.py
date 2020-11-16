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

from IPython.display import HTML

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
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['week'] = df.index.week
    df['weekday'] = df.index.weekday
    df['daylight'] = ((df['hour'] >= 7) & (df['hour'] <= 19)).astype(int)
    df['weekend'] = ((df['weekday'] >= 5) & (df['weekday'] <= 6)).astype(int)
    #df.set_index('Timestamp', drop=True, inplace=True)
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

def get_outliers(df, Modules, k_factor=1.5, std_times=3, verbose=True, remove=True, method='z_score'):

    '''
    This function identifies outliers in a Dataset.
    Two methods can be used, z_score and interquartile.
    If the data follos a Gaussian like distribution Z_score
    can be used, if not is better to use the interquartile method.
    This are not very good estimation since some of this outliers can be
    noveltys (correct data that explains something that happens
    or viceversa)
    k_factor: factor for interquartile method
    std_times: how many standard deviations away from the mean
               for Z_Score method
    df: Original Dataframe
    remove: If False don't remove the outliers
    Modules: Dictionary type
    '''

    # Dataframe to save Outliers by Modules
    df_outliers = pd.DataFrame(data=None)

    for module in Modules:
        # Take a slice of the dataset by module
        slice_of_data = df[df['Module']== module]
        # Drop Useless Columns (columns with all nan values)
        slice_of_data = slice_of_data.dropna(axis=1, how='all')
        if method == 'z_score':
            # calculate summary statistics
            mean, std = slice_of_data['Temp_Mod'].mean().round(4), slice_of_data['Temp_Mod'].std().round(4)
            # identify outliers outside 3 standar deviations
            cut_off = std * std_times
            lower, upper = mean - cut_off, mean + cut_off
            # identify outliers
            outliers = [x for x in slice_of_data['Temp_Mod'] if x < lower or x > upper]
            # save outliers in dataframe
            df_outliers = pd.concat([df_outliers, pd.Series(outliers, name=module, dtype=float)], axis=1)
            #print(Counter(outliers))
            outliers_removed = [x for x in slice_of_data['Temp_Mod'] if x > lower and x < upper]

            if verbose == True:
                print('Outliers indentified: {} in module {}'.format(len(outliers), module))
                print('Non-outlier observations: {} in module {}'.format(len(outliers_removed), module))

        if method == 'interquartile':
            q25, q75 = np.percentile(slice_of_data['Temp_Mod'], 25), np.percentile(slice_of_data['Temp_Mod'], 75)
            IQR = q75 - q25
            cut_off = IQR * k_factor
            lower, upper = q25 - cut_off, q75 + cut_off
            outliers = [x for x in slice_of_data['Temp_Mod'] if x < lower or x > upper]
            outliers_removed = [x for x in slice_of_data['Temp_Mod'] if x >= lower and x <= upper]
            # save outliers in dataframe
            df_outliers = pd.concat([df_outliers, pd.Series(outliers, name=module, dtype=float)], axis=1)

            if verbose == True:
                print('Percentiles 25th = {} 75th = {} IQR = {}'.format(q25, q75, IQR.round(3)))
                print('Outliers indentified {}'.format(len(outliers)))
                print('Non-outlier observations: %d' % len(outliers_removed))


        # Remove the Outliers
        if remove == True:
            for ii in np.unique(outliers):
                if verbose == True:
                    print('Removing {} from Module {}'.format(ii, module))
                else:
                    slice_of_data = df[df['Module']== module]
                    indexx = slice_of_data[slice_of_data['Temp_Mod'] == ii].index
                    df = df.drop(df.index[indexx], axis=0)
                    df.reset_index(inplace=True, drop=True)

    return df_outliers, df

def create_train_test(slice_of_data, resample_time='5Min', feature_index=0, feature_name='Temp_Mod', scaler=StandardScaler(),
                      final_date='2019-01-20 00:00:05', split_date = '2019-01-01 00:00:05', test_date = '2019-01-17 00:00:05',
                      figsize=(20,4)):
    '''
    The passed dataset must have a timestamp as index.
    Resamples the dataset into a given freq (default 5 min) using
    rolling mean and split it into training and testing dataframes.
    Scales the data using Sklearn Standard Scaler and returns the
    numpy arrays of the x_train, y_train, x_test, y_test.

    slice_of_data: data to be splitted
    feature_index: index of the feature to be predicted (for y_train and test)
    feature_name: name of the feature in dataset
    final_date: date limit to split
    split_date: date to start splitting
    test_date: starting test date

    data: dataframe with all the resampled data, combines train and test
    '''

    # Resample to 5 min with rolling mean
    slice_of_data = slice_of_data.resample(resample_time).mean()
    # Method Backward Fill
    slice_of_data = slice_of_data.fillna(slice_of_data.bfill())

    data = slice_of_data.loc[split_date:final_date]
    train_data = slice_of_data.loc[split_date:test_date]
    test_data = slice_of_data.loc[test_date:final_date]

    #Scale the data to unit variance. We only fit in the Training data
    #scaler = StandardScaler()
    # Scale the data and convert it into dataframe for easy splitting
    X_train = pd.DataFrame(scaler.fit_transform(train_data), columns=[slice_of_data.columns]).to_numpy()
    X_test = pd.DataFrame(scaler.transform(test_data), columns=[slice_of_data.columns]).to_numpy()

    # Since i'm going to predict only temperature
    y_train, y_test = X_train[:,feature_index], X_test[:,feature_index]

    slice_of_data.loc[split_date:test_date][feature_name].plot(figsize=figsize, label='Train')
    slice_of_data.loc[test_date:final_date][feature_name].plot(figsize=figsize, title=('Resampled Data to {} per data (Rolling mean)'.format(resample_time)), color='r', label='Test')
    plt.legend(); plt.tight_layout()

    return data, train_data, test_data, X_train, X_test, y_train, y_test, scaler

def separete_modules(df, dictionary):
    '''
    Create a list of dataframes with different modules
    '''
    slices = []
    for module in dictionary:
        # Take a slice of the dataset by module
        # Drop Useless Columns (columns with all nan values)
        slice_of_data = df[df['Module']== module].dropna(axis=1, how='all')
        slice_of_data.reset_index(inplace=True, drop=True)
        slices.append(slice_of_data)
    return np.asarray(slices)

def serialize_model(model, history, name='model'):
    '''
    Save model and history
    '''
    # serialize model to JSON
    model_json = model.to_json()
    with open(name+'.json', "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(name+'.h5')
    history.to_csv(name+'.csv')
    print("Saved model to disk")

def load_model(name='model'):
    json_file = open(name+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(name+'.h5')
    print("Loaded model from disk")
    return loaded_model

def split_sequences(X_train, feature_index=0, n_steps=32):

    '''
    Split a multivariate sequence into samples for single feature prediction
    Taken and adapted from Machinelearningmastery.
    Split the training set into segments of a specified timestep
    and creates the labels.
    '''
    n_steps = n_steps+1
    # Place the column of the feature to predict at the end of the dataset
    sequences = np.concatenate([X_train, X_train[:,feature_index].reshape(-1,1)],axis=1)

    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix-1, :-1], sequences[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)

    print(np.shape(X),np.shape(y))
    return np.array(X), np.array(y)

def split_sequences_multivariate(sequences, n_steps=32):

    '''
    Split a multivariate sequence into samples for single feature prediction
    Taken and adapted from Machinelearningmastery.
    Split the training set into segments of a specified timestep
    and creates the labels.
    '''
    #n_steps = n_steps+1
    # Place the column of the feature to predict at the end of the dataset
    #sequences = np.concatenate([X_train, X_train[:,0].reshape(-1,1)],axis=1)

    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)

    print(np.shape(X),np.shape(y))
    return np.array(X), np.array(y)

def get_abs_err(X, y, model, std=3, plot=True, n_outputs=6):
    '''
    get the absolute error between prediction and labels
    and the threshold. Plots the error distribution also.
    '''
    x_pred = model.predict(X)

    if np.shape(y)[-1] == n_outputs:
        abs_err = np.asarray(abs(x_pred - y))
        threshold = abs_err[0,:].std()*std
        if plot==True:
            fig, (ax1) = plt.subplots(1, 1, figsize=(10,4))
            sns.distplot(abs_err[0,:], ax=ax1, label='Error distribution')
            plt.axvline(x=threshold, ymin=0, ymax=8, color='r', label='Threshold')
            plt.title('Error distribution, threshold set in {}'. format(threshold.round(4)))
            #plt.xticks(np.arange(0,1.1,0.1))
            plt.legend(); plt.tight_layout()
            return pd.DataFrame(abs_err), threshold
        else:
            return pd.DataFrame(abs_err)
    else:
        abs_err = np.asarray(abs(x_pred - y.reshape(-1,1)))
        threshold = abs_err.std()*std
        if plot==True:
            fig, (ax1) = plt.subplots(1, 1, figsize=(10,4))
            sns.distplot(abs_err, ax=ax1, label='Error distribution')
            plt.axvline(x=threshold, ymin=0, ymax=8, color='r', label='Threshold')
            plt.title('Error distribution, threshold set in {}'. format(threshold.round(4)))
            #plt.xticks(np.arange(0,1.1,0.1))
            plt.legend(); plt.tight_layout()
            return pd.Series(abs_err.reshape(-1)), threshold
        else:
            return pd.Series(abs_err.reshape(-1))

def create_segments(data, length=50):
    '''
    create sequence previous data points for each data points
    in this case y is one dimensional because my output is going to be one value alone
    '''
    result = []
    for index in range(len(data) - length):
        result.append(data[index: index + length])
    return np.asarray(result)

def plot_model_results(model, history_df, X, y, index, feature_index=0):
    '''
    Plot the scores of the model and the prediction vs the
    real time series
    '''
    prediction = model.predict(X)
    if prediction.shape[-1] == 1:
        pred_df = pd.DataFrame(data= np.concatenate((model.predict(X), y.reshape(-1,1)), axis=1), columns=['Prediction', 'True'], index=index)
    else:
        pred_df = pd.DataFrame(data= np.concatenate((model.predict(X)[:,feature_index].reshape(-1,1), y[:,feature_index].reshape(-1,1)), axis=1), columns=['Prediction', 'True'], index=index)

    fig = plt.figure(figsize=(24,5))
    spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[1, 3])
    ax0 = fig.add_subplot(spec[0])
    sns.lineplot(data=history_df[['loss','val_loss']], markers=True)
    # plot the prediction and the reality (for the test data)
    ax1 = fig.add_subplot(spec[1])
    pred_df.plot(ax=ax1)
    plt.legend(loc='upper left')
    plt.title('Prediction and Truth (Scaled)')
    plt.tight_layout()

def detect_anomaly(data, threshold, error, train_shape):

    df = data
    df.reset_index(inplace=True)
    # data with anomaly label (test data part)
    test = (error >= threshold).astype(int)
    complement = pd.Series(0, index=np.arange(train_shape))

    # add the data to the main
    df['anomaly_LSTM'] = complement.append(test, ignore_index='True')
    df.set_index('Timestamp',inplace=True)
    #print(df['anomaly_LSTM'].value_counts())
    anomaly = df.loc[data['anomaly_LSTM'] == 1, ['Temp_Mod']] #anomaly
    return df, anomaly

def get_anomaly_and_pred(model, X, y, threshold, test_data, n_features=6, feature=0):

    index=test_data.index[64:]
    if np.shape(y)[-1] == n_features:
        predictions = np.concatenate((model.predict(X)[:,feature].reshape(-1,1), y[:,feature].reshape(-1,1)), axis=1)
    else:
        predictions = np.concatenate((model.predict(X), y.reshape(-1,1)), axis=1)

    predictions_df = pd.DataFrame(data=predictions, columns=['Pred','True'], index=index)
    predictions_df['error'] = abs(predictions_df['True'] - predictions_df['Pred'])

    dates_index = predictions_df['error'].loc[predictions_df.error >= threshold].index

    anomaly = pd.Series(data=np.zeros(shape=(test_data.shape[0])), index=test_data.index, name='anomalies')
    anomaly.loc[dates_index] = predictions_df['error'].loc[predictions_df.error >= threshold].values.astype(bool)
    #print('Anomalies founded {}'.format(anomaly.value_counts()[1]))

    return predictions_df, anomaly

def plot_anomalies_model(data, test_date, anomalies, select='LSTM',anomalies_svm=None, anomalies_forest=None):

    # data and anomaly are dataframes
    fig = plt.figure(figsize=(24,4))
    spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[1, 3])

    if select=='LSTM':
        ax0 = fig.add_subplot(spec[0])
        sns.distplot(a=data[test_date:]['Temp_Mod'], kde=False, ax=ax0, label='Data')
        sns.distplot(a=data[data['anomaly_LSTM']==1]['Temp_Mod'], kde=False, color='darkred', ax=ax0, label='LSTM')
        ax0.legend()

        ax1 = fig.add_subplot(spec[1])
        sns.lineplot(data[test_date:].index, data[test_date:]['Temp_Mod'], label='Data', ax=ax1)
        ax1.scatter(data[data['anomaly_LSTM']==1].index, data[data['anomaly_LSTM']==1]['Temp_Mod'], color='red', s=30, label='LSTM anomalies')
        ax1.set_title('Anomalies detected in the last days')
        ax1.legend()

    if select == 'all':
        ax0 = fig.add_subplot(spec[0])
        sns.distplot(a=data[test_date:]['Temp_Mod'], kde=False, ax=ax0, label='Data')
        sns.distplot(a=data[data['anomaly_LSTM']==1]['Temp_Mod'], kde=False, color='darkred', ax=ax0, label='LSTM')
        sns.distplot(a=anomalies_svm[test_date:]['Temp_Mod'], kde=False, color='orange', ax=ax0, label='SVM')
        sns.distplot(a=anomalies_forest[test_date:]['Temp_Mod'], kde=False, color='green', ax=ax0, label='Isolation Forest')
        ax0.legend()

        ax1 = fig.add_subplot(spec[1])
        sns.lineplot(data[test_date:].index, data[test_date:]['Temp_Mod'], ax=ax1, label='Data')
        ax1.scatter(anomalies[anomalies==1].index, data[data['anomaly_LSTM']==1]['Temp_Mod'], color='red', s=30, label='LSTM anomalies')
        ax1.scatter(anomalies_svm[test_date:].index, anomalies_svm[test_date:]['Temp_Mod'], color='black', s=40, label='SVM', marker='x')
        ax1.scatter(anomalies_forest[test_date:].index, anomalies_forest[test_date:]['Temp_Mod'], color='orange', s=40, label='Isolation Forest')
        ax1.set_title('Anomalies detected in the last days')
        ax1.legend()

    if select=='others':
        ax0 = fig.add_subplot(spec[0])
        sns.distplot(a=data['Temp_Mod'], kde=False, ax=ax0)
        sns.distplot(a=anomalies['Temp_Mod'], kde=False, color='darkred', ax=ax0)
        ax1 = fig.add_subplot(spec[1])
        sns.lineplot(data.index, data['Temp_Mod'], ax=ax1)
        ax1.scatter(anomalies.index, anomalies['Temp_Mod'], color='red', s=20)

    plt.tight_layout()

def isolation_forest(data, n_estimators=50, outliers_fraction=0.01, scaler=StandardScaler()):

    df = data.copy()
    index = data.index
    df.reset_index(inplace=True, drop=True)

    #scaler = StandardScaler() #Scale the data to unit variance. We only fit in the Training data
    train = pd.DataFrame(scaler.fit_transform(df), columns=[df.columns]) # Scale the data and convert it into dataframe for easy splitting

    # train isolation forest
    model =  IsolationForest(contamination=outliers_fraction, n_estimators=50, max_samples='auto', max_features=1.0)
    model.fit(train)

    # add the data to the main
    df['scores_isolation_f'] = pd.Series(model.decision_function(train))
    df['anomaly_IsolationF'] = pd.Series(model.predict(train))
    df['anomaly_IsolationF'] = df['anomaly_IsolationF'].map( {1: 0, -1: 1} )
    #print(df['anomaly_IsolationF'].value_counts())

    df = df.set_index(index, drop=True)
    # anomalies marked as 1
    anomaly_Iforest = df.loc[df['anomaly_IsolationF'] == 1, ['Temp_Mod']] #anomaly

    return df, anomaly_Iforest

def oneClass_SVM(data, nu=0.95, outliers_fraction=0.01, scaler=StandardScaler()):

    df = data.copy()
    index = data.index
    df.reset_index(inplace=True, drop=True)

    #scaler = StandardScaler() #Scale the data to unit variance. We only fit in the Training data
    train = pd.DataFrame(scaler.fit_transform(df), columns=[df.columns]) # Scale the data and convert it into dataframe for easy splitting

    # train isolation forest
    model =  OneClassSVM(nu=nu * outliers_fraction) #nu=0.95 * outliers_fraction  + 0.05
    model.fit(train)

    # add the data to the main
    df['anomaly_SVM'] = pd.Series(model.predict(train))
    df['anomaly_SVM'] = df['anomaly_SVM'].map( {1: 0, -1: 1} )

    df = df.set_index(index, drop=True)
    # anomalies marked as 1
    anomaly_svm = df.loc[df['anomaly_SVM'] == 1, ['Temp_Mod']] #anomaly

    return df, anomaly_svm

def forecast_LSTM(model, X_train, X_test, n_timesteps, n_features):
    test_predictions = []
    first_eval_batch = X_train[-n_timesteps:]
    current_batch = first_eval_batch.reshape((1, n_timesteps, n_features))

    for i in range(len(X_test)):

        # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array])
        current_pred = model.predict(current_batch)[0]

        # store prediction
        test_predictions.append(current_pred)

        # update batch to now include prediction and drop first value
        current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)

    return test_predictions

    RADIOS, RECEIVERS, POWERS = get_radios_rec_powers(df)
