# Copyright (c) 2019 ETH Zurich

import os, sys
import numpy as np
import time
import re
import argparse
import configparser
import ast
from rdkit import Chem

sys.path.append('../src/')
from python import helper as hp
from python import helper_chem as hp_chem
from python import fixed_parameters as FP

parser = argparse.ArgumentParser(description='Run scores analysis')
parser.add_argument('-fn','--filename', type=str, help='Path to the fine-tuning txt file', required=True)
parser.add_argument('-v','--verbose', type=bool, help='Verbose', required=True)
args = vars(parser.parse_args())


if __name__ == '__main__':
    
    start = time.time()
    
    ####################################
    # get back parameters
    args = vars(parser.parse_args())
    
    verbose = args['verbose']
    filename = args['filename']
    name_data = filename.split('/')[-1].replace('.txt','')
    config = configparser.ConfigParser()
    config.read('parameters.ini')
    
    if verbose: print('\nSTART COMPUTING DESCRIPTORS, SCAFFOLDS AND FINGERPRINTS')
    ####################################
   
    
    
    
    ####################################
    # path to the saved novo data
    path_novo = f'results/{name_data}/novo_molecules/'
    
    # path to save the scores
    save_path = f'results/{name_data}/analysis/'
    os.makedirs(save_path, exist_ok=True)
    ####################################
    
    
    

    ####################################
    # start iterating over the files
    for filename in os.listdir(path_novo):
        if filename.endswith('.pkl'):
            name = filename.replace('.pkl', '')
            data = hp.load_obj(f'{path_novo}{name}')
            novo_tr = data['novo_tr']
            
            # We do a double check because rdkit 
            # throws weird errors sometimes
            data_rdkit = []
            for i,x in enumerate(novo_tr):
                mol = Chem.MolFromSmiles(x)
                if mol is not None:
                    data_rdkit.append(mol)
            
            save_name = name.split('_')[1] + '_' + name.split('_')[2]
            
            # descriptors
            desc_names = re.compile(FP.DESCRIPTORS['names'])
            functions, names = hp_chem.get_rdkit_desc_functions(desc_names)
            desc_dict = hp_chem.rdkit_desc(data_rdkit, functions, names)
            hp.save_obj(desc_dict, save_path + f'desc_{save_name}')
            
            # scaffolds
            scaf, generic_scaf = hp_chem.extract_murcko_scaffolds(data_rdkit)
            desc_scaf = {'scaffolds': scaf, 'generic_scaffolds': generic_scaf}
            hp.save_obj(desc_scaf, f'{save_path}scaf_{save_name}')
            hp.write_in_file(f'{save_path}{save_name}_scaffolds.txt', scaf)
            hp.write_in_file(f'{save_path}{save_name}_generic_scaffolds.txt', generic_scaf)
            
            
            # fingerprints
            fingerprint = hp_chem.fingerprint_calc(data_rdkit, verbose=verbose)
            fp_dict = {'fingerprint': fingerprint}
            hp.save_obj(fp_dict, save_path + f'fp_{save_name}')
            
            
    end = time.time()
    if verbose: print(f'EXTRACTING DESCRIPTORS, SCAFFOLDS AND FINGERPRINTS DONE in {end - start:.04} seconds')
    ####################################
                    
                