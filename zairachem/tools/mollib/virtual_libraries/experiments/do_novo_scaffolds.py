# Copyright (c) 2019 ETH Zurich

import os, sys
import time
import argparse
import configparser
import ast
from rdkit import Chem
from rdkit import rdBase
rdBase.DisableLog('rdApp.*')

sys.path.append('../src/')
from python import helper as hp

parser = argparse.ArgumentParser(description='Run novo analysis for scaffolds')
parser.add_argument('-fn','--filename', type=str, help='Path to the fine-tuning txt file', required=True)
parser.add_argument('-s','--scaffolds_type', type=str, help='Type of the scaffolds. generic_scaffolds or scaffolds.', required=True)
parser.add_argument('-v','--verbose', type=bool, help='Verbose', required=True)


if __name__ == '__main__':
    
    start = time.time()
    
    ####################################
    # get back parameters
    args = vars(parser.parse_args())
    
    verbose = args['verbose']
    scaffolds_type = args['scaffolds_type']
    filename = args['filename']
    name_data = filename.split('/')[-1].replace('.txt','')
    config = configparser.ConfigParser()
    config.read('parameters.ini')
    
    # get back the experiment parameters
    min_len = int(config['PROCESSING']['min_len'])
    max_len = int(config['PROCESSING']['max_len'])
    mode = config['EXPERIMENTS']['mode']
    
    if verbose: print(f'\nSTART NOVO SCAFFOLDS ANALYSIS FOR {scaffolds_type}')
    ####################################
    
    
    
    
    ####################################         
    # path to the saved generated data
    path_scaf = f'results/{name_data}/analysis/'

        
    # path to save the novo analysis
    save_path = f'results/{name_data}/novo_{scaffolds_type}/'
    os.makedirs(save_path, exist_ok=True)
    ####################################
    
    
    
    
    ####################################
    # load the source space scaffolds
    # i.e. from the dataset used for 
    # pretraining the model
    def get_back_scaffolds(space, scaffolds_type):
        name = config['DATA'][space]
        name = name.replace('.txt','')
        aug = int(config['AUGMENTATION'][space])
        path_to_scaf = f'results/data/{name}/{min_len}_{max_len}_x{aug}/scaf'
        
        if not os.path.isfile(f'{path_to_scaf}.pkl'):
            raise ValueError('Scaffolds cannot be accessed. Are they missing for the source or taget space?')
        else:
            scaf = hp.load_obj(path_to_scaf)[scaffolds_type]
        
        return scaf
    
    scaf_source = get_back_scaffolds('source_space', scaffolds_type)
    
    # load the target set (from which the 
    # fine-tuning set comes from) scaffolds
    # if provided
    if config['DATA']['target_space']:
        scaf_target = get_back_scaffolds('target_space', scaffolds_type)
    ####################################
    
    
    
    
    ####################################
    # scaffolds of the fine-tuning 
    # molecules
    if mode == 'fine_tuning':
        aug = int(config['AUGMENTATION']['fine_tuning'])
        path_to_scaf = f'results/data/{name_data}/{min_len}_{max_len}_x{aug}/scaf'
        scaf_ft = hp.load_obj(path_to_scaf)[scaffolds_type]
    ####################################  
    
    
    
    
    ####################################
    # start iterating over the files
    t0 = time.time()
    for filename in os.listdir(path_scaf):
        if filename.endswith('.pkl') and 'scaf' in filename:
            name = filename.replace('.pkl', '')
            data = hp.load_obj(f'{path_scaf}{name}')[scaffolds_type]
                        
            checked_scaf = []
            n_valid = 0
            
            for gen_scaf in data:
                if len(gen_scaf)!=0 and isinstance(gen_scaf, str):
                    mol = Chem.MolFromSmiles(gen_scaf)
                    if mol is not None: 
                        cans = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
                        if len(cans)>=1:
                            checked_scaf.append(cans)
                            n_valid+=1
            
            novo_analysis = {}
            
            if len(checked_scaf)>0:
                unique_set = set(checked_scaf)
                n_unique = len(unique_set)
                novo_tr_source = list(unique_set - set(scaf_source))
                n_novo_tr_source = len(novo_tr_source)
                
                if config['DATA']['target_space']:
                    novo_tr_target = list(unique_set - set(scaf_target))
                    n_novo_tr_target = len(novo_tr_target)
                    
                    novo_tr_both = list(unique_set - set(scaf_target) - set(scaf_source))
                    n_novo_tr_both = len(novo_tr_both)
                    
                    novo_analysis['n_novo_tr_both'] = n_novo_tr_both
                    novo_analysis['n_novo_tr_target'] = n_novo_tr_target
                    
                if mode=='fine_tuning':
                    novo_tr_ft = list(unique_set - set(scaf_ft))
                    n_novo_tr_ft = len(novo_tr_ft)
                
                    novo_analysis['n_novo_tr_ft'] = n_novo_tr_ft

                
                novo_analysis['n_valid'] = n_valid
                novo_analysis['n_unique'] = n_unique
                novo_analysis['n_novo_tr_source'] = n_novo_tr_source
                    
                # save
                novo_name = save_path + name.replace('scaf', scaffolds_type)
                hp.save_obj(novo_analysis, novo_name)
                
                # we save the novo scaffolds also as a .txt
                with open(f'{novo_name}_source.txt', 'w+') as f:
                    for item in novo_tr_source:
                        f.write("%s\n" % item)
                
                if verbose: print(f'scaffolds analysis for {name} done')
            else:
                print(f'There is n {n_valid} valid scaffolds for {name}')
                
    
    end = time.time()    
    if verbose: print(f'NOVO SCAFFOLDS ANALYSIS FOR {scaffolds_type} DONE in {end - start:.04} seconds')
    ####################################  
    