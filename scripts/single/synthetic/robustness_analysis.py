from config import conf
from conformal_prediction import ConformalPrediction
import os
from expert.expert import ExpertSynthetic
import pickle
import utils
from model.model import ModelSynthetic
from tqdm import tqdm
import numpy as np

"""Robustness analysis for near alpha value in synthetic experiments"""

sensitivity_prob = np.arange(0.05, 1.05, 0.05)

results_root = f"{conf.ROOT_DIR}/results_synthetic"
if not os.path.exists(results_root):
    os.mkdir(results_root)

results_root+=f"/{conf.n_labels}labels_calibrationSet{conf.cal_split}"
if not os.path.exists(results_root):
    os.mkdir(results_root)

# Compute empirical expert misprediction probability for near optimal alpha for all human and model combinations 
# given the number of labels, and calibration and estimation split, under IIA violations
for human_accuracy in conf.accuracies:
    for machine_accuracy in conf.accuracies:
        for run in tqdm(range(conf.n_runs_per_split)):

            res_dir = f"{results_root}/human{human_accuracy}_machine{machine_accuracy}_run{run}"
            if not os.path.exists(res_dir):
                os.mkdir(res_dir)
            
            # Generate dataset
            X_train, X_test, X_cal, X_est, y_train, y_test, y_cal, y_est = utils.make_dataset(run, machine_accuracy)
            
            # Human expert
            conf.accuracy = human_accuracy
            human = ExpertSynthetic(conf)

            # Create model
            model = ModelSynthetic()
            model.train(X_train, y_train)
            print(model.test(X_test, y_test))

            conf_pred = ConformalPrediction(X_cal, y_cal, X_est, y_est, model, conf.delta)
            
            # Get alpha values
            alphas_dir = f"{res_dir}/alphas1"
            if os.path.exists(alphas_dir):
                with open(alphas_dir, 'rb') as f:
                    alphas = pickle.load(f)
            else:
                alphas = conf_pred.find_all_alpha_values()
                with open(alphas_dir, 'wb') as f1:
                    pickle.dump(alphas, f1, pickle.HIGHEST_PROTOCOL)

            # Get index of near optimal alpha value
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
            
            # error in test set under IIA violations
            for p in sensitivity_prob:
                error = conf_pred.test_error_robustness(p=p, X_test=X_test, y_test=y_test, w_matrix=human.w_matrix, alpha1_value=alpha1_value).detach().cpu().numpy()
                robustness_error_dict[p] = error
            
            robustness_dict_dir = f"{res_dir}/alpha1_robustness_error_dict"
            with open(robustness_dict_dir, 'wb') as f:
                pickle.dump(robustness_error_dict, f, pickle.HIGHEST_PROTOCOL)
