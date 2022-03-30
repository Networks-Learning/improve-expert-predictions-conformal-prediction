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



results_root = f"{conf.ROOT_DIR}/results_real"
if not os.path.exists(results_root):
    os.mkdir(results_root)

results_root+=f"/calibrationSet{conf.cal_split}"
if not os.path.exists(results_root):
    os.mkdir(results_root)



for model_name in conf.model_names:
    for run in tqdm(range(conf.n_runs_per_split)):

        res_dir = f"{results_root}/{model_name}_run{run}"
        if not os.path.exists(res_dir):
            os.mkdir(res_dir)
      
       
        X_test, X_cal, X_est, y_test, y_cal, y_est = utils.make_dataset_real(run)
                

        conf.accuracy = None
        human = ExpertReal(conf)

        model = ModelReal(model_name)

        conf_pred = StandardCPgpu(X_cal, y_cal,X_est, y_est, model, conf.delta)
        with open(f"{res_dir}/alphas1",'rb') as f1:
            alphas =  pickle.load(f1)
        size_t = conf_pred.size_given_test_set_per_a(X_test,y_test, human.w_matrix,alphas)
        size = size_t.detach().cpu().numpy()
        with open(f"{res_dir}/set_size_test",'wb') as f2:
            pickle.dump(size, f2 ,pickle.HIGHEST_PROTOCOL)            
                   
                   






















