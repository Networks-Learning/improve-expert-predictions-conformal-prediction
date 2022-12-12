from config import conf
from conformal_prediction import ConformalPrediction
import numpy as np
import os
from expert.expert import ExpertReal
import utils
from model.model import ModelReal
from tqdm import tqdm
import pickle

"""Script for robustness analysis for the near optimal alpha value in real data experiments"""

sensitivity_prob = np.arange(0.05, 1.05, 0.05)

results_root = f"{conf.ROOT_DIR}/results_real"
if not os.path.exists(results_root):
    os.mkdir(results_root)

results_root+=f"/calibrationSet{conf.cal_split}"
if not os.path.exists(results_root):
    os.mkdir(results_root)

# Compute empirical expert misprediction probability for near optimal alpha for all models 
# given the calibration and estimation split, under IIA violations
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

        conf_pred = ConformalPrediction(X_cal, y_cal, X_est, y_est, model, conf.delta)

        # Get alpha values
        alphas_dir = f"{res_dir}/alphas1"
        if os.path.exists(alphas_dir):
            with open(alphas_dir, 'rb') as f1:
                alphas = pickle.load(f1)
        else:
            alphas = conf_pred.find_all_alpha_values()
            with open(alphas_dir, 'wb') as f1:
                pickle.dump(alphas, f1, pickle.HIGHEST_PROTOCOL)

        # Get index of near optimal alpha
        alpha_hat_dir = f"{res_dir}/alpha1_idx"
        if os.path.exists(alpha_hat_dir):
            with open(alpha_hat_dir, 'rb') as f1:
                alpha_hat_idx = pickle.load(f1)
        else:
            alpha_hat_idx = conf_pred.find_a_star(human.w_matrix)
            with open(alpha_hat_dir, 'wb') as f1:
                pickle.dump(alpha_hat_idx, f1, pickle.HIGHEST_PROTOCOL)

        alpha1_value = alphas[alpha_hat_idx]
        robustness_error_dict = {}
        
        # Error in test set under IIA violations 
        for p in sensitivity_prob:
            error = conf_pred.test_error_robustness(p=p, X_test=X_test, y_test=y_test, w_matrix=human.w_matrix, alpha1_value=alpha1_value).detach().cpu().numpy()
            robustness_error_dict[p] = error
        
        robustness_dict_dir = f"{res_dir}/alpha1_robustness_error_dict"
        with open(robustness_dict_dir, 'wb') as f:
            pickle.dump(robustness_error_dict, f, pickle.HIGHEST_PROTOCOL)
        