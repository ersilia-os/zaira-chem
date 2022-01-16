# Copyright (c) 2019 ETH Zurich

import os, sys
import time
import re
import argparse
import configparser
import ast
import glob
import pandas as pd
import matplotlib
matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('../src/')
from python import helper as hp
from python import fixed_parameters as FP

parser = argparse.ArgumentParser(description='Do descriptor plot')
parser.add_argument('-fn','--filename', type=str, help='Path to the fine-tuning txt file', required=True)
parser.add_argument('-v','--verbose', type=bool, help='Verbose', required=True)


def do_combined_boxplot(df, desc_to_plot, epochs_to_plot, save_path, te):

    fig, ax = plt.subplots(figsize=(12,8))
    
    start_palette = ['#1575A4']
    epoch_palette = ['#F5F5F5']*len(epochs_to_plot)
    end_palette = ['#5A5A5A', '#D55E00']
    sns.boxplot(x="seq_time", y="value", data=df, palette=start_palette+epoch_palette+end_palette, 
                showfliers=False, width=0.35)
    sns.despine(offset=15, trim=True)
            
    tick_font_sz = FP.PAPER_FONT['tick_font_sz']
    label_font_sz = FP.PAPER_FONT['label_font_sz']
    legend_sz = FP.PAPER_FONT['legend_sz']
    
    ax.set_ylim(0,1)
    plt.xlabel('')
    plt.ylabel(desc_to_plot, fontsize=label_font_sz)
    
    #+3 for: src space, tgt space and transfer learning set
    ax.set_xticks([y for y in range(len(epochs_to_plot)+3)])
    start_xticklabels = ['Source space']
    epoch_xticklabels = [f'epoch {e}' for e in epochs_to_plot]
    end_xticklabels = ['Transfer\nlearning\nset', 'Target space']
    
    ax.set_xticklabels(start_xticklabels + epoch_xticklabels + end_xticklabels, 
                       rotation=30,
                       fontsize=tick_font_sz)
    
    plt.yticks(fontsize=tick_font_sz)
    plt.savefig(f'{save_path}{desc_to_plot}_{te}.png', bbox_inches='tight')

def update_df(df, dict_temp):
    df_temp = pd.DataFrame.from_dict(dict_temp)
    frames = [df, df_temp]
    df = pd.concat(frames)
    return df

def get_dict_with_data(data_name, min_len, max_len, aug):
    data_des = hp.load_obj(f'results/data/{data_name}/{min_len}_{max_len}_x{aug}/desc.pkl')
    des = data_des[desc_to_plot]
    dict_temp = {'seq_time': [data_name]*len(des),
                 'value': des}
    return dict_temp
    
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
    
    temp = float(config['EXPERIMENTS']['temp'])
    min_len = int(config['PROCESSING']['min_len'])
    max_len = int(config['PROCESSING']['max_len'])
    desc_to_plot = FP.DESCRIPTORS['names']  
    desc_to_plot = re.search(r'\((.*?)\)', desc_to_plot).group(1)
    
    if verbose: print('\nSTART DESCRIPTOR PLOT')
    ####################################
    
    
    
    
    ####################################        
    # Path to the descriptors
    path_des = f'results/{name_data}/analysis/'
    
    # Path to save the novo analysis
    save_path = f'results/{name_data}/plot_descriptor/'
    os.makedirs(save_path, exist_ok=True)
    ####################################
    
    
    
    
    ####################################
    # get back data
    df = pd.DataFrame(columns=['seq_time', 'value'])

    # get back dataset descriptor
    src_space_name = config['DATA']['source_space']
    src_space_name = src_space_name.replace('.txt','')
    dict_temp = get_dict_with_data(src_space_name, min_len, max_len, 
                                   int(config['AUGMENTATION']['source_space']))
    df = update_df(df, dict_temp)
    
    for fname in sorted(os.listdir(path_des)):
        if fname.endswith('.pkl'):
            if 'desc' in fname and str(temp) in fname:
                name = fname.replace('.pkl', '')
                epoch = int(name.split('_')[1])
                seq_time = f'epoch {epoch}'
                    
                # get values 
                data = hp.load_obj(path_des + fname)
                values = data[desc_to_plot]
                
                # add to dataframe
                dict_temp = {'seq_time': [seq_time]*len(values),
                             'value': values}
                df = update_df(df, dict_temp)
                
    dict_temp = get_dict_with_data(name_data, min_len, max_len, 
                                   int(config['AUGMENTATION']['fine_tuning']))
    df = update_df(df, dict_temp)
                
    tgt_space_name = config['DATA']['target_space']
    tgt_space_name = tgt_space_name.replace('.txt','')
    dict_temp = get_dict_with_data(tgt_space_name, min_len, max_len, 
                                   int(config['AUGMENTATION']['target_space']))
    df = update_df(df, dict_temp)
    
    # we get back the epoch sampled from the saved models
    all_models = glob.glob(f'results/{name_data}/models/*.h5')
    epochs_to_plot = sorted([x.split('/')[-1].replace('.h5', '') for x in all_models], key=int)
    do_combined_boxplot(df, desc_to_plot, epochs_to_plot, save_path, temp)
                
    end = time.time()
    if verbose: print(f'DESCRIPTOR PLOT DONE in {end - start:.04} seconds')
    ####################################
        