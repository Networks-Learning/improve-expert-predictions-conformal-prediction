from config import conf
from conformal_prediction import StandardCPgpu
import numpy as np
import os
from expert.expert import ExpertReal, ExpertSynthetic
import pickle
import utils
from model.model import ModelReal, ModelSynthetic
import sys
import datetime
from tqdm import tqdm


"""Script for getting the average set size per alpha value in real data experiments"""

results_root = f"{conf.ROOT_DIR}/results_real"
if not os.path.exists(results_root):
    os.mkdir(results_root)

results_root+=f"/calibrationSet{conf.cal_split}"
if not os.path.exists(results_root):
    os.mkdir(results_root)

# Compute average set size per alpha for all human and model combinations 
# given the calibration and estimation split
for model_name in conf.model_names:
    for run in tqdm(range(conf.n_runs_per_split)):

        res_dir = f"{results_root}/{model_name}_run{run}"
        if not os.path.exists(res_dir):
            os.mkdir(res_dir)
      
        # Read and split dataset
        X_test, X_cal, X_est, y_test, y_cal, y_est = utils.make_dataset_real(run)
                
        # Human expert
        conf.accuracy = None
        human = ExpertReal(conf)
        
        # Create model
        model = ModelReal(model_name)

        conf_pred = StandardCPgpu(X_cal, y_cal,X_est, y_est, model, conf.delta)

        # Get alpha values
        alphas_dir = f"{res_dir}/alphas1"
        if os.path.exists(alphas_dir):
            with open(alphas_dir,'rb') as f1:
                alphas =  pickle.load(f1)
        else:
            alphas = conf_pred.find_all_alpha_values()
            with open(alphas_dir,'wb') as f1:
                pickle.dump(alphas, f1 ,pickle.HIGHEST_PROTOCOL)
    

        size_t = conf_pred.size_given_test_set_per_a(X_test,y_test, human.w_matrix,alphas)
        size = size_t.detach().cpu().numpy()
        with open(f"{res_dir}/set_size_test",'wb') as f2:
            pickle.dump(size, f2 ,pickle.HIGHEST_PROTOCOL)            
                   
                