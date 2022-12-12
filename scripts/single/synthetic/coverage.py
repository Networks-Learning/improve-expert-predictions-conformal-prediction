from config import conf
from conformal_prediction import ConformalPrediction
import os
from expert.expert import ExpertSynthetic
import pickle
import utils
from model.model import ModelSynthetic
from tqdm import tqdm

"""Script for empirical coverage in synthetic experiments"""

results_root = f"{conf.ROOT_DIR}/results_coverage_synthetic"
if not os.path.exists(results_root):
    os.mkdir(results_root)

results_root+=f"/{conf.n_labels}labels_calibrationSet{conf.cal_split}"
if not os.path.exists(results_root):
    os.mkdir(results_root)

machine_accuracy = 0.5
human_accuracy = 0.5

for run in tqdm(range(conf.n_runs_per_split)):

    res_dir = f"{results_root}/human{human_accuracy}_machine{machine_accuracy}_run{run}"
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)
    
    # Create dataset
    X_train, X_test, X_cal, X_est, y_train, y_test, y_cal, y_est = utils.make_dataset(run, machine_accuracy)

    # Human expert
    conf.accuracy = human_accuracy
    human = ExpertSynthetic(conf)

    # Create machine
    model = ModelSynthetic()
    model.train(X_train, y_train)

    conf_pred = ConformalPrediction(X_cal, y_cal, X_est, y_est, model, conf.delta)

    # Get alpha values
    alphas_dir = f"{res_dir}/alphas1"
    if os.path.exists(alphas_dir):
        with open(alphas_dir, 'rb') as f:
            alphas = pickle.load(f)
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

    # Compute empirical coverage in test set
    empirical_cov_t = conf_pred.empirical_coverage(X_test, y_test, alpha_hat)
    empirical_cov = empirical_cov_t.detach().cpu().numpy()
    with open(f"{res_dir}/alpha1_emp_coverage", 'wb') as f1:
        pickle.dump(empirical_cov, f1, pickle.HIGHEST_PROTOCOL)
