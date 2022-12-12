from config import conf
from conformal_prediction import ConformalPrediction
import os
from expert.expert import ExpertSynthetic
import pickle
import utils
from model.model import ModelSynthetic
import numpy as np
import sys
import datetime
from tqdm import tqdm

"""Script for sythetic data experiments for the shifted quantile method"""

original_stdout = sys.stdout
original_stderr = sys.stderr

results_root = f"{conf.ROOT_DIR}/results_synthetic_modified_cp"
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
                try :
                    with open(f"{res_dir}/logs.txt", 'w', buffering=1) as f:
                        sys.stdout = f
                        
                        print(f"Creating {conf.data_size} data:"+ \
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

                        print(f"{now()}: Finding alpha values...")
                        conf_pred = ConformalPrediction(X_cal, y_cal ,X_est , y_est, model, conf.delta)
                        alphas = conf_pred.find_all_alpha_values()
                        with open(f"{res_dir}/alphas1", 'wb') as f1:
                            pickle.dump(alphas, f1, pickle.HIGHEST_PROTOCOL)
                        
                        print(f"{now()}: Shifted quantile method...")
                        best_success_prob_est_set = 0
                        best_a1a2 = (0, 1)
                        best_a1 = 0
                        best_success_prob_est_set_conf_pred = 0           

                        for alpha1_idx, alpha1 in enumerate(alphas):
                            
                            alpha2_idx, success_prob_est_set, success_prob_est_set_conf_pred = conf_pred.find_a_star(human.w_matrix, a1_star_idx=alpha1_idx, all_a1_a2=True)
                            
                            if success_prob_est_set > best_success_prob_est_set:
                                best_success_prob_est_set = success_prob_est_set
                                alpha2 = np.append(alphas[alphas > alpha1], 1)[alpha2_idx]
                                best_a1a2 = (alpha1, alpha2)
                            
                            if success_prob_est_set_conf_pred > best_success_prob_est_set_conf_pred:
                                best_a1 = alpha1
                                best_success_prob_est_set_conf_pred = success_prob_est_set_conf_pred
                        
                        print(f"{now()}: Done. Saving results...")
                        with open(f"{res_dir}/alpha1_value", 'wb') as f1:
                            pickle.dump(best_a1a2[0], f1, pickle.HIGHEST_PROTOCOL)
                        
                        with open(f"{res_dir}/alpha2_value", 'wb') as f1:
                            pickle.dump(best_a1a2[1], f1, pickle.HIGHEST_PROTOCOL)

                        print(f"{now()}: Calculating error in test set for near optimal alpha1, alpha2")
                        p_error_t = conf_pred.error_given_test_set_given_a(X_test, y_test, human.w_matrix, best_a1a2[0], best_a1a2[1])
                        p_error = p_error_t.detach().cpu().numpy()
                
                        with open(f"{res_dir}/alpha1_alpha2_test_error", 'wb') as f1:                
                            pickle.dump(p_error, f1, pickle.HIGHEST_PROTOCOL)

                        print(f"{now()}: Calculating error in test set for near optimal alpha1")
                        p_error_t1 = conf_pred.error_given_test_set_given_a(X_test, y_test, human.w_matrix, best_a1)
                        p_error1 = p_error_t1.detach().cpu().numpy()
                
                        with open(f"{res_dir}/alpha1_test_error", 'wb') as f1:                
                            pickle.dump(p_error1, f1, pickle.HIGHEST_PROTOCOL)

                        sys.stdout = original_stdout
                    sys.stderr = original_stderr
                except:
                    print(sys.exc_info(), file=f_e)
                    sys.stdout = original_stdout
                    sys.stderr = original_stderr
                    raise