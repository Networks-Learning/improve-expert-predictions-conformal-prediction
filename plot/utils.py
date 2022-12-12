import numpy as np
import pandas as pd
from expert.expert import ExpertReal, ExpertRealMoreExpressive
import pickle
from config import conf
import torch
import utils
import os

def real_models_acc(split, runs=10):
    """Test set accuracy of pre-trained models"""
    base = f"results_real/calibrationSet{split}"
    d = []
    for machine_model in reversed(conf.model_names):
        best = []
        for run in range(runs):
            dir = f"{base}/{machine_model}_run{run}"
            with open(f'{dir}/logs.txt', 'r') as f:
                acc = float(f.readlines()[3])
                d.append((machine_model, acc))

    df = pd.DataFrame(d, columns=["Model", "Accuracy"])
    
    return df.groupby("model").mean()

def real_human_acc(split, run=0, has_groups=False, empirical=True):
    """Human test set accuracy in real data experiments"""
    conf.cal_split = split
    if not has_groups:
        X_test, X_cal, X_est, y_test, y_cal, y_est = utils.make_dataset_real(run)                
    else:
        X_test, X_cal, X_est, y_test, y_cal, y_est, y_groups = utils.make_dataset_real_with_difficulties(run)                
    
    conf.accuracy = None
    if not has_groups:
        human = ExpertReal(conf)
    else:
        human = ExpertRealMoreExpressive(conf, y_groups)

    if not has_groups:
        p = torch.tensor(human.confusion_matrix[y_test], device=conf.device)
    else:
        p = torch.tensor(human.confusion_matrix[y_test[:,1], y_test[:,0]], device=conf.device)

    y_hat = p.multinomial(1, replacement=True, generator=conf.torch_rng).detach().cpu().numpy().flatten() 
    
    if not has_groups:
        if empirical:
            return 1 - np.mean((y_test != y_hat))
        else:
            return np.diag(human.confusion_matrix).mean()
    else:
        if empirical:
            return 1 - np.mean((y_test[:,0] != y_hat))
        else:
            return list(np.diag(human.confusion_matrix[i]).mean() for i in set(y_groups))

def read_specific_data(root_dir, keys):
    ret_dict = {}
    for root,dirs,files in os.walk(root_dir):
        for name in keys:
            with open(f"{root}/{name}", "rb") as f:
                val = pickle.load(f)
                ret_dict[name] = val

    return ret_dict

def get_synthetic_results(base_dir, keys, runs, fn, human_accs=conf.accuracies, machine_accs=conf.accuracies):
    entries_to_return = []
    for human_accuracy in human_accs: 
        for machine_accuracy in machine_accs:
            for run in range(runs):
                dir = f"{base_dir}/human{human_accuracy}_machine{machine_accuracy}_run{run}"
                data = read_specific_data(dir, keys)
                res_tuple = fn(data,human_accuracy)
                entries_to_return.append((human_accuracy, machine_accuracy)+res_tuple+(run,))
    return entries_to_return

def get_real_results(base_dir, keys, runs, fn):
    entries_to_return = []
    for machine_model in reversed(conf.model_names):
        for run in range(runs):
            dir = f"{base_dir}/{machine_model}_run{run}"                       
            data = read_specific_data(dir, keys)
            res_tuple = fn(data, run)
            entries_to_return.append((machine_model,)+res_tuple+(run,))
    return entries_to_return
