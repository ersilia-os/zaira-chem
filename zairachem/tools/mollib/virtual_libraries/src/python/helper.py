# Copyright (c) 2019 ETH Zurich

import sys, os
import pickle
import pandas as pd

def read_with_pd(path, delimiter='\t', header=None):
    data_pd = pd.read_csv(path, delimiter=delimiter, header=header)
    return data_pd[0].tolist() 

def save_obj(obj, name):
    """save obj with pickle"""
    name = name.replace('.pkl', '')
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    """load a pickle object"""
    name = name.replace('.pkl', '')
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def write_in_file(path_to_file, data):
    with open(path_to_file, 'w+') as f:
        for item in data:
            f.write("%s\n" % item)
            
                              
                
                

