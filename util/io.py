import os
import ast
import csv
import glob
import pickle
import scipy.io
import numpy as np
import pandas as pd
from tqdm import tqdm
from biosppy.signals import tools
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer


def compute_label_aggregations(df, folder):

    df['scp_codes_len'] = df.scp_codes.apply(lambda x: len(x))

    aggregation_df = pd.read_csv(folder+'scp_statements.csv', index_col=0)

    df['all_scp'] = df.scp_codes.apply(lambda x: list(set(x.keys())))

    return df


def load_dataset(path, sampling_rate, release=False):

    # load and convert annotation data
    Y = pd.read_csv(path+'icbeb_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load raw signal data
    X = load_raw_data_icbeb(Y, sampling_rate, path)

    return X, Y


def select_data(XX, YY, min_samples, outputfolder):
    # convert multilabel to multi-hot
    mlb = MultiLabelBinarizer()

    # filter 
    counts = pd.Series(np.concatenate(YY.all_scp.values)).value_counts()
    counts = counts[counts > min_samples]
    YY.all_scp = YY.all_scp.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
    YY['all_scp_len'] = YY.all_scp.apply(lambda x: len(x))
    # select
    X = XX[YY.all_scp_len > 0]
    Y = YY[YY.all_scp_len > 0]
    mlb.fit(Y.all_scp.values)
    y = mlb.transform(Y.all_scp.values)
   
    # save LabelBinarizer
    with open(outputfolder+'mlb.pkl', 'wb') as tokenizer:
        pickle.dump(mlb, tokenizer)

    return X, Y, y, mlb


def load_raw_data_icbeb(df, sampling_rate, path):

    if sampling_rate == 100:
        if os.path.exists(path + 'raw100.npy'):
            data = np.load(path+'raw100.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path + 'records100/'+str(f)) for f in tqdm(df.index)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path+'raw100.npy', 'wb'), protocol=4)
    elif sampling_rate == 500:
        if os.path.exists(path + 'raw500.npy'):
            data = np.load(path+'raw500.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path + 'records500/'+str(f)) for f in tqdm(df.index)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path+'raw500.npy', 'wb'), protocol=4)
            
    return data


def load_formmated_raw_data(inputfolder, outputfolder, sampling_frequency=500):

    # Load data
    data, raw_labels = load_dataset(inputfolder, sampling_frequency)
    
    # Preprocess label data
    labels = compute_label_aggregations(raw_labels, inputfolder)
        
    # Select relevant data and convert to one-hot
    data, labels, Y, _ = select_data(
        data, labels, min_samples=0, outputfolder=outputfolder)
    
    return data, Y, labels, _.classes_


def load_snippet_data(inputfolder):

    pickle_in = open(inputfolder, "rb")

    data = pickle.load(pickle_in)

    X = data['data']

    Y = data['label']

    I = data['index']

    L = data['length']

    return X, Y, I, L


def load_state_data(inputfolder):

    data_dict = load_pkfile(inputfolder)

    return data_dict


def load_csv(filepath):
    data = []
    with open(filepath, newline='') as csvfile:
        spamreader = csv.reader(csvfile,delimiter=',',quotechar = '|')
        #next(spamreader)
        for row in spamreader:
            data.append(row[0].split(":"))
    return data


def check_folder(path):

    if not os.path.exists(path):
        try:
            os.makedirs(path)
            print("Create : ", path)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    else:
        print(path, " exists")


def load_pkfile(inputfolder):

    pickle_in = open(inputfolder, "rb")

    data_in = pickle.load(pickle_in)

    pickle_in.close()

    return data_in


def save_pkfile(outputfolder, data):

    pickle_out = open(outputfolder, "wb")

    pickle.dump(data, pickle_out)

    pickle_out.close()

    print(outputfolder, "saving successful !")






