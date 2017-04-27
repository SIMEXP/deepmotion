import numpy as np
import h5py

def write(dict_data, file_name, compress=4):
    with h5py.File(file_name, 'w') as hf:
        for ii in range(len(dict_data.keys())):
            item_id = dict_data.keys()[ii]
            hf.create_dataset(item_id, data=dict_data[item_id], compression="gzip", compression_opts=compress)

def load(file_name):
    with h5py.File(file_name, 'r') as hf:
        dict_data = {}
        print("List of arrays in this file: ", hf.keys())
        for ii in range(len(hf.keys())):
            item_id = hf.keys()[ii]
            dict_data[item_id] = np.array(hf.get(item_id))

    return dict_data