from config import conf
from conformal_prediction import ConformalPrediction
import numpy as np
import os
from expert.expert import ExpertSynthetic
import utils
from model.model import ModelSynthetic
from tqdm import tqdm
import pickle

"""Script for getting the set size distribution for the near optimal alpha value in synthetic data experiments"""

results_root = f"{conf.ROOT_DIR}/results_synthetic"
if not os.path.exists(results_root):
    os.mkdir(results_root)

results_root+=f"/{conf.n_labels}labels_calibrationSet{conf.cal_split}"
if not os.path.exists(results_root):
    os.mkdir(results_root)

# Compute set size distribution for near optimal alpha for all combinations of humans and machines 
# given the calibration and estimation split
for human_accuracy in conf.accuracies:
    for machine_accuracy in conf.accuracies:
        for run in tqdm(range(conf.n_runs_per_split)):

            res_dir = f"{results_root}/human{human_accuracy}_machine{machine_accuracy}_run{run}"

            if not os.path.exists(res_dir):
                os.mkdir(res_dir)
        
            # Generate dataset
            X_train, X_test, X_cal, X_est, y_train, y_test, y_cal, y_est  = utils.make_dataset(run, machine_accuracy)
                    
            # Human expert
            conf.accuracy = human_accuracy
            human = ExpertSynthetic(conf)
            
            # Create model
            model = ModelSynthetic()
            model.train(X_train, y_train)

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

            # Set-size distribution for near optimal alpha
            alpha1 = alphas[alpha_hat_idx]
            sizes_t, counts_t = conf_pred.size_given_test_set_given_a(X_test, alpha1)
            sizes = sizes_t.detach().cpu().numpy()
            counts = counts_t.detach().cpu().numpy()
            set_size_distr = {s:c for s,c in zip(sizes,counts)}
            
            with open(f"{res_dir}/alpha1_set_size_distr_dict", 'wb') as f2:
                pickle.dump(set_size_distr, f2, pickle.HIGHEST_PROTOCOL)
          