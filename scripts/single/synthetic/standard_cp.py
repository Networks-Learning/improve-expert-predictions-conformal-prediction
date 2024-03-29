from config import conf
from conformal_prediction import ConformalPrediction
import numpy as np
import os
from expert.expert import ExpertSynthetic
import pickle
import utils
from model.model import ModelSynthetic
import sys
import datetime
from tqdm import tqdm

"""Script for synthetic experiments"""

original_stdout = sys.stdout
original_stderr = sys.stderr

results_root = f"{conf.ROOT_DIR}/results_synthetic"
if not os.path.exists(results_root):
    os.mkdir(results_root)

results_root+=f"/{conf.n_labels}labels_calibrationSet{conf.cal_split}"
if not os.path.exists(results_root):
    os.mkdir(results_root)

now = lambda:datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# For a given number of labels and calibration and estimation split 
# run experiments for all combinations of humans and machines
for human_accuracy in conf.accuracies:
    for machine_accuracy in conf.accuracies:
        for run in tqdm(range(conf.n_runs_per_split)):

            res_dir = f"{results_root}/human{human_accuracy}_machine{machine_accuracy}_run{run}"
            if not os.path.exists(res_dir):
                os.mkdir(res_dir)
            with open(f"{res_dir}/logs_err.txt", 'w', buffering=1) as f_e:
                sys.stderr = f_e
                try:
                    with open(f"{res_dir}/logs.txt", 'w', buffering=1) as f:
                        sys.stdout = f
                        
                        print(f"Creating {conf.data_size} data:"+\
                              f"{(1 - conf.cal_split - conf.test_split)*100}% train, "+\
                              f"{conf.cal_split*100}% calibration, "+\
                              f"{conf.test_split*100}% test")
                        X_train, X_test, X_cal, X_est, y_train, y_test, y_cal, y_est = utils.make_dataset(run, machine_accuracy)

                        print(f"Creating human with {human_accuracy*100}% accuracy")
                        conf.accuracy = human_accuracy
                        human = ExpertSynthetic(conf)

                        print(f"Creating machine with {machine_accuracy*100}% accuracy")
                        model = ModelSynthetic()
                        model.train(X_train, y_train)
                        print(model.test(X_test, y_test))

                        print(f"{now()}: Starting conformal prediction...")
                        conf_pred = ConformalPrediction(X_cal, y_cal, X_est, y_est, model, conf.delta)
                        alphas = conf_pred.find_all_alpha_values()
                        with open(f"{res_dir}/alphas1", 'wb') as f1:
                            pickle.dump(alphas, f1, pickle.HIGHEST_PROTOCOL)
                        
                        print(f"{now()}: {alphas.shape[0]} alphas found")
                        alpha_star_idx = conf_pred.find_a_star(human.w_matrix)
                        with open(f"{res_dir}/alpha1_idx", 'wb') as f1:
                            pickle.dump(alpha_star_idx, f1, pickle.HIGHEST_PROTOCOL)

                        alpha1_value = conf_pred.alphas[alpha_star_idx]
                        print(f"Near-optimal alpha = {alpha1_value}")

                        with open(f"{res_dir}/alpha1_value", 'wb') as f1:
                            pickle.dump(alpha1_value, f1, pickle.HIGHEST_PROTOCOL)

                        print(f"{now()}: Calculating error in test set for all alphas")
                        p_error_t = conf_pred.error_given_test_set_per_a(X_test, y_test, human.w_matrix, alphas)
                        p_error = p_error_t.detach().cpu().numpy()
                        with open(f"{res_dir}/alpha1_test_error", 'wb') as f1:
                            pickle.dump(p_error, f1, pickle.HIGHEST_PROTOCOL)

                        sys.stdout = original_stdout
                    sys.stderr = original_stderr
                except:
                    print(sys.exc_info(), file=f_e)
                    sys.stdout = original_stdout
                    sys.stderr = original_stderr
                    raise


















