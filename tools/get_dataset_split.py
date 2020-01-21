import numpy as np
import argparse

import os
import glob

import json

np.random.seed(0)

def get_split(datapoints_list, split_name, train_percentage=0.8):
    if split_name == 'image_wise':
        num_datapoints = len(datapoints_list)

    train_size = int(num_datapoints * train_percentage)

    np.random.shuffle(datapoints_list)
    
    train_list = datapoints_list[:train_size]
    val_list = datapoints_list[train_size:]

    return train_list, val_list

def write_split_file(dataset_path, split_name, train_percentage=0.8):
    datapoints_list = [os.path.basename(x) for x in glob.glob(os.path.join(dataset_path, 'datapoints', '*.np*'))]
    train_list, val_list = get_split(datapoints_list, split_name, train_percentage)

    with open(os.path.join(dataset_path, 'train_split.json'), 'w') as outfile:
        json.dump(train_list, outfile)

    with open(os.path.join(dataset_path, 'val_split.json'), 'w') as outfile:
        json.dump(val_list, outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str, default=None, help='name of folder to save the training dataset in')
    parser.add_argument('--split_name', type=str, default='image_wise', help='how to split the dataset. default: image-wise')
    args = parser.parse_args()
    dataset_path = args.dataset_path
    split_name = args.split_name

    if dataset_path is None:
        assert False # pls provide the dataset path

    write_split_file(dataset_path, split_name)

    

    
