import scipy.io
from tqdm import tqdm
import pickle
import glob
import argparse
import os
import numpy as np
import cv2
import re


# The diagnoses list from the paper in SNOMED format -> [SR, MI, LAD, LVH, AF, STach, IAVB, SB, TAb]
LABELS_LIST = ['426783006', '164865005', '39732003', '164873001', '164889003', '427084000', '270492004', '426177001', '164934002']


def transform_frequency(signal, freq, sec):
    resized = cv2.resize(np.array(signal), (freq*sec, signal.shape[0]))
    resized = resized.astype(signal.dtype)
    return resized
    

def get_labels(filenames):
    heads_labels = []
    for filename in tqdm(filenames):
        with open(filename, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if 'Dx' in line:
                line = line.strip()
                heads_labels.append(re.findall(r'\d+', line))
    return heads_labels


def encode_labels(labels, l2id):
    labels_encoded = []
    for head in labels:
        res_i = [0]*len(l2id)
        for head_i in head:
            if head_i in l2id:
                res_i[l2id[head_i]] = 1
        labels_encoded.append(res_i)
    labels_encoded = np.array(labels_encoded)
    return labels_encoded


def prepare_signals(output_path, dataset_name, signals, seq_len):
    # Signals preparing
    values = []
    for signal in tqdm(signals):
        values.append(scipy.io.loadmat(signal)['val'])
    for i, val in enumerate(values):
        values[i] = transform_frequency(val, seq_len, 10)
    values = np.array(values)      
    # Normalization
    res_min, res_max = [], []
    for ax in tqdm([0,2,6,7,8,9,10,11]):
        res_min.append(np.percentile(values[:,ax,:].min(axis=1), 2.5))
        res_max.append(np.percentile(values[:,ax,:].max(axis=1), 97.5))
    res_min = np.array(res_min)[None, :, None]
    res_max = np.array(res_max)[None, :, None]
    
    data_path = os.path.join(output_path, dataset_name)
    os.makedirs(data_path, exist_ok=True)    
    
    os.makedirs(os.path.join(data_path, "thirdparty"), exist_ok=True)
    np.save(os.path.join(data_path, "thirdparty", "res_min.npy"), res_min)
    np.save(os.path.join(data_path, "thirdparty", "res_max.npy"), res_max)
    min_good = (values[:, [0,2,6,7,8,9,10,11], :] > res_min).all(axis=(1,2))
    max_good = (values[:, [0,2,6,7,8,9,10,11], :] < res_max).all(axis=(1,2))
    # Filtered ECGs indexes select
    ids = np.arange(0, len(values))
    ids = ids[min_good & max_good]
    return ids, values[ids]


def split_data(data_path, values_count):
    # Train / test split
    shuffled_ids = np.arange(values_count)
    np.random.seed(42)
    np.random.shuffle(shuffled_ids)
    os.makedirs(os.path.join(data_path, "thirdparty"), exist_ok=True)
    train_ids = shuffled_ids[shuffled_ids.shape[0]//5:]
    np.save(os.path.join(data_path, "thirdparty", "train_ids.npy"), train_ids)
    test_val_ids = shuffled_ids[:shuffled_ids.shape[0]//5]
    test_ids = test_val_ids[test_val_ids.shape[0]//2:]
    np.save(os.path.join(data_path, "thirdparty", "test_ids.npy"), test_ids)
    val_ids = test_val_ids[:test_val_ids.shape[0]//2]
    np.save(os.path.join(data_path, "thirdparty", "val_ids.npy"), val_ids)   


def prepare_data(dataset_path, output_path, seq_len):
    heads_ptbxl, heads_georgia, heads_ningbo = np.array(glob.glob(os.path.join(dataset_path, "ptb-xl", "*/*.hea"))), np.array(glob.glob(os.path.join(dataset_path, "georgia", "*/*.hea"))), np.array(glob.glob(os.path.join(dataset_path, "ningbo", "*/*.hea")))
    signals_ptbxl, signals_georgia, signals_ningbo = np.array(list(map(lambda x: x.replace("hea", "mat"), heads_ptbxl))), np.array(list(map(lambda x: x.replace("hea", "mat"), heads_georgia))), np.array(list(map(lambda x: x.replace("hea", "mat"), heads_ningbo)))
    # Get good signals ids
    ids_ptbxl, signals_ptbxl = prepare_signals(output_path, "ptb-xl", signals_ptbxl, seq_len)
    ids_georgia, signals_georgia = prepare_signals(output_path, "georgia", signals_georgia, seq_len)
    ids_ningbo, signals_ningbo = prepare_signals(output_path, "ningbo", signals_ningbo, seq_len)
    np.save(os.path.join(output_path, "ptb-xl", "signals.npy"), signals_ptbxl)
    np.save(os.path.join(output_path, "georgia", "signals.npy"), signals_georgia)
    np.save(os.path.join(output_path, "ningbo", "signals.npy"), signals_ningbo)
    # Names filtering
    heads_ptbxl = heads_ptbxl[ids_ptbxl]
    heads_georgia = heads_georgia[ids_georgia]
    heads_ningbo = heads_ningbo[ids_ningbo]
    # Labels obtaining
    labels_ptbxl = get_labels(heads_ptbxl)
    labels_georgia = get_labels(heads_georgia)
    labels_ningbo = get_labels(heads_ningbo)
    # Forming label2id dict
    label2id, i = {}, 0
    for label in LABELS_LIST:
        label2id[label] = i
        i += 1
    with open(os.path.join(output_path, 'label2id.pickle'), 'wb') as handle:
        pickle.dump(label2id, handle, protocol=pickle.HIGHEST_PROTOCOL) 

    # Labels encoding
    encoded_ptbxl = encode_labels(labels_ptbxl, label2id)
    encoded_georgia = encode_labels(labels_georgia, label2id)
    encoded_ningbo = encode_labels(labels_ningbo, label2id)

    assert encoded_ptbxl.shape[0] == signals_ptbxl.shape[0]
    assert encoded_georgia.shape[0] == signals_georgia.shape[0]
    assert encoded_ningbo.shape[0] == signals_ningbo.shape[0]

    np.save(os.path.join(output_path, "ptb-xl", "labels.npy"), encoded_ptbxl)
    np.save(os.path.join(output_path, "georgia", "labels.npy"), encoded_georgia)
    np.save(os.path.join(output_path, "ningbo", "labels.npy"), encoded_ningbo)
    # Data split
    split_data(os.path.join(output_path, "ptb-xl"),len(encoded_ptbxl))
    split_data(os.path.join(output_path, "georgia"), len(encoded_georgia))
    split_data(os.path.join(output_path, "ningbo"), len(encoded_ningbo))

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Obtain data from 2021 Challenge.')
    parser.add_argument('dataset_path', type=str)
    parser.add_argument('save_path', type=str)
    parser.add_argument('--input_size', type=int, default=500)

    args = parser.parse_args()
    prepare_data(args.dataset_path, args.save_path, args.input_size)