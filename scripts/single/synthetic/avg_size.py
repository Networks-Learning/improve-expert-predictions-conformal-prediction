from config import conf
from conformal_prediction import ConformalPrediction
import os
from expert.expert import ExpertSynthetic
import pickle
import utils
from model.model import ModelSynthetic
from tqdm import tqdm

"""Script for getting the empirical average set size per alpha value in synthetic experiments"""

results_root = f"{conf.ROOT_DIR}/results_synthetic"
if not os.path.exists(results_root):
    os.mkdir(results_root)

results_root+=f"/{conf.n_labels}labels_calibrationSet{conf.cal_split}"
if not os.path.exists(results_root):
    os.mkdir(results_root)

# Compute average set size per alpha for all human and model combinations 
# given the number of labels, and calibration and estimation split
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
            
            size_t = conf_pred.size_given_test_set_per_a(X_test, alphas)
            size = size_t.detach().cpu().numpy()
            with open(f"{res_dir}/alpha1_avg_set_size_test", 'wb') as f2:
                pickle.dump(size, f2, pickle.HIGHEST_PROTOCOL)
 