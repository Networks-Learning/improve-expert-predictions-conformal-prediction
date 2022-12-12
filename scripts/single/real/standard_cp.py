from config import conf
from conformal_prediction import ConformalPrediction
import os
from expert.expert import ExpertReal
import pickle
import utils
from model.model import ModelReal
import sys
import datetime
from tqdm import tqdm

"""Script for real data experiments"""

original_stdout = sys.stdout
original_stderr = sys.stderr

results_root = f"{conf.ROOT_DIR}/results_real"
if not os.path.exists(results_root):
    os.mkdir(results_root)

results_root+=f"/calibrationSet{conf.cal_split}"
if not os.path.exists(results_root):
    os.mkdir(results_root)

now = lambda:datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# For a given number of calibration and estimation split 
# run experiments for all models
for model_name in conf.model_names:
    for run in tqdm(range(conf.n_runs_per_split)):

        res_dir = f"{results_root}/{model_name}_run{run}"
        if not os.path.exists(res_dir):
            os.mkdir(res_dir)
        with open(f"{res_dir}/logs_err.txt", 'w', buffering=1) as f_e:
            sys.stderr = f_e
            try:
                with open(f"{res_dir}/logs.txt", 'w', buffering=1) as f:
                    sys.stdout = f
                    
                    print(f"Creating {conf.data_size} data: "+\
                          f"{conf.cal_split*100}% calibration, "+\
                          f"{(1 - conf.cal_split)*100}% test")
                    X_test, X_cal, X_est, y_test, y_cal, y_est = utils.make_dataset_real(run)
                
                    print(f"Initializing human ")
                    conf.accuracy = None
                    human = ExpertReal(conf)

                    print(f"Initializing model ")
                    model = ModelReal(model_name)
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
                    print(f"alpha_1* {conf_pred.alphas[alpha_star_idx]}")

                    print(f"{now()}: Calculating error in test set for all alphas")
                    p_error_t = conf_pred.error_given_test_set_per_a(X_test, y_test, human.w_matrix, alphas)
                    p_error = p_error_t.detach().cpu().numpy()
                    with open(f"{res_dir}/alpha1_test_error", 'wb') as f1:
                        pickle.dump(p_error, f1, pickle.HIGHEST_PROTOCOL)

                    sys.stdout = original_stdout
                sys.stderr = original_stderr    
            except:
                print(sys.exc_info()[0], file=f_e)
                sys.stdout = original_stdout
                sys.stderr = original_stderr
                raise






















