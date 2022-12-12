from config import conf
from conformal_prediction import ConformalPrediction
import os
from expert.expert import ExpertReal
import utils
from model.model import ModelReal
from tqdm import tqdm
import pickle


"""Script for getting the empirical coverage for the near optimal alpha value in real data experiments"""

results_root = f"{conf.ROOT_DIR}/results_coverage_real"
if not os.path.exists(results_root):
    os.mkdir(results_root)

results_root+=f"/calibrationSet{conf.cal_split}"
if not os.path.exists(results_root):
    os.mkdir(results_root)

# Compute empirical coverage for near optimal alpha  
model_name = conf.model_names[0]
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
        
    # Get index of near optimal alpha
    alpha_hat_dir = f"{res_dir}/alpha1_value"
    if os.path.exists(alpha_hat_dir):
        with open(alpha_hat_dir, 'rb') as f1:
            alpha_hat = pickle.load(f1)
    else:
        alpha_hat_idx = conf_pred.find_a_star(human.w_matrix)
        with open(f"{res_dir}/alpha1_idx", "wb") as f:
            pickle.dump(alpha_hat_idx, f, pickle.HIGHEST_PROTOCOL)

        alpha_hat = alphas[alpha_hat_idx]
        with open(alpha_hat_dir, 'wb') as f1:
            pickle.dump(alpha_hat, f1, pickle.HIGHEST_PROTOCOL)
        
    #Compute empirical coverage in test set
    empirical_cov_t = conf_pred.empirical_coverage(X_test, y_test, alpha_hat)
    empirical_cov = empirical_cov_t.detach().cpu().numpy()
    with open(f"{res_dir}/alpha1_emp_coverage", 'wb') as f1:
        pickle.dump(empirical_cov, f1, pickle.HIGHEST_PROTOCOL)
