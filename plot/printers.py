from os import error
import numpy as np
import pandas as pd
from config import conf
from plot.utils import *

pm_char = u"\u00B1"

def accuracy_synthetic(split=.15, labels=10, results_root='results_synthetic', 
                       runs=10, latex=False, method='standard'):
    """Expert success probability using our optimized system for each
       expert and classification task given a calibration set split 
       and a label space in the synthetic data experiments"""
       
    def get_success(data, keys=[], method=0):
        # conformal prediction method
        if method=='standard':
            alpha1_idx, alpha1_test_error = tuple(data[k] for k in keys)
            return (1 - alpha1_test_error[alpha1_idx],)
        else:
            alpha1_alpha2_test_error = data[keys[0]]
            return (1 - alpha1_alpha2_test_error,)

    base = f"{results_root}/{labels}labels_calibrationSet{split}"
    if method=='modified':
        keys = ['alpha1_alpha2_test_error']
    else:
        keys = ['alpha1_idx','alpha1_test_error']

    entries = get_synthetic_results(base, keys=keys, runs=runs, fn=lambda x,y: get_success(x,keys=keys,method=method))

    df = pd.DataFrame(entries, columns=["Human", "Machine", "Accuracy", "runs"])
    pvt_df = df.pivot_table(index='Human', columns="Machine", values="Accuracy")
    pvt_df_std = df.pivot_table(index='Human', columns="Machine", values="Accuracy", aggfunc=np.std)/np.sqrt(runs)

    annot = (pvt_df).round(2).astype("string") 
    annot_std = (pvt_df_std).round(2).astype("string") # std error

    if latex:
        print(annot.to_latex())
    else:
        return annot+pm_char+annot_std   

def gain_over_labels_and_cal_data(splits=[0.02,0.05,0.1,0.15], spaces=[10,50,100], runs=10, results_root='results_synthetic'):
    """Relative gain in expert success probability for 
       all splits and number of labels in synthetic experiments """
    entries = []
    for split in splits:
        for labels in spaces:
            base = f"{results_root}/{labels}labels_calibrationSet{split}"
            n_data = int((conf.data_size* (1 - conf.test_split))*split)
            keys = ['alpha1_idx', 'alpha1_test_error']
            get_gain = lambda data, human: (100*(1 - data['alpha1_test_error'][data['alpha1_idx']] - human) / human ,)

            gains = get_synthetic_results(base, keys, runs=runs, fn=get_gain)
            to_extend = [(n_data, labels) + g for g in gains]
            entries.extend(to_extend)
            
    succ_p = 'Success probability gain'
    df = pd.DataFrame(entries, columns=['m', 'n', 'Human Acc', 'Machine_acc', succ_p, 'run'])
    
    mean_df_mn = pd.pivot_table(df, index='m', columns='n', values=succ_p).round(3).astype("string")
    std_df_mn = (pd.pivot_table(df, index='m', columns='n', values=succ_p, aggfunc=np.std)/(np.sqrt((len(conf.accuracies)**2)*runs))).round(3).astype("string")
    
    return  mean_df_mn+pm_char+std_df_mn 

def accuracy_real(split=0.15, runs=10, results_root='results_real', latex=False):
    """Expert success probability using our optimized system for each
       classifier given a calibration set split in the real data experiments"""

    base_dir  = f"{results_root}/calibrationSet{split}"
    keys = ['alpha1_idx', 'alpha1_test_error']

    get_acc = lambda data, run: (1 - data['alpha1_test_error'][data['alpha1_idx']],)

    entries = get_real_results(base_dir=base_dir, keys=keys, runs=runs, fn=get_acc)

    df = pd.DataFrame(entries, columns=["Machine", "Accuracy", "runs"])
    pvt_df = df.pivot_table(index="Machine", values="Accuracy")
    pvt_df_std = df.pivot_table(index="Machine", values="Accuracy", aggfunc=np.std)/np.sqrt(runs)

    annot = pvt_df.round(3).astype("string") 
    annot_std = pvt_df_std.round(3).astype("string") # std error

    if latex:
        print(annot.to_latex())
    else:
        return annot+pm_char+annot_std   

def gain_over_cal_data_real(splits=[0.02, 0.05, 0.1,0.15], runs=10, results_root="results_real"):
    """Relative gain in success probability for all splits in real data experiments"""
    entries = []
    def get_gain(data, split=0.0, run=0):
        human_accuracy = real_human_acc(split=split, run=run)        
        alpha_hat_error = data['alpha1_test_error'][data['alpha1_idx']]
        gain = (100*(1 - alpha_hat_error  - human_accuracy) / human_accuracy)
        return gain

    keys = ['alpha1_idx', 'alpha1_test_error']

    for split in splits:
        base = f"{results_root}/calibrationSet{split}"
        n_data = int(conf.data_size*split)
        gains = get_real_results(base, keys, runs, lambda data,run : (get_gain(data, split=split, run=run),))
        to_extend = [(n_data, ) + g for g in gains]
        entries.extend(to_extend)
    
    succ_p = 'Success probability gain%'
    df = pd.DataFrame(entries, columns=['m', 'machine', succ_p, 'run'])
    mean_df_m = pd.pivot_table(df, index='m', values=succ_p).round(2).astype("string")
    std_df_m = (pd.pivot_table(df, index='m', values=succ_p, aggfunc=np.std)/np.sqrt(len(conf.model_names)*runs)).round(2).astype("string")

    return mean_df_m+pm_char+std_df_m
 
def alpha2_less_than_one(split=.02, labels=10, runs=10, results_root='results_synthetic_modified_cp'): 
    """Fraction of times near optimal alpha_2 < 1 for each expert in each synthetic task"""
    base  = f"{results_root}/{labels}labels_calibrationSet{split}"
    keys = ['alpha2_value']
    get_alpha2s = lambda data,_ : (int(data[keys[0]] < 1),)
    entries = get_synthetic_results(base, keys, runs, get_alpha2s)
    
    df = pd.DataFrame(entries, columns=["Human", "Machine", "alpha2 < 1", 'runs' ])
    pvt_df = df.pivot_table(index='Human', columns="Machine", values="alpha2 < 1")
    pvt_df_std = df.pivot_table(index='Human', columns="Machine", values="alpha2 < 1", aggfunc=np.std)/np.sqrt(runs)
    
    annot = (pvt_df).round(2).astype("string") 
    annot_std = (pvt_df_std).round(2).astype("string") 

    return annot+pm_char+annot_std
