from config import conf, args
from conformal_prediction import ConformalPrediction
import os
from expert.expert import ExpertSynthetic
import pickle
import utils
from model.model import ModelSynthetic
import sys
import datetime
from tqdm import tqdm

"""Script for synthetic experiments with experts using a top-k predictor"""

original_stdout = sys.stdout
original_stderr = sys.stderr
k = args.topk

results_root = f"{conf.ROOT_DIR}/results_synthetic_top{k}"
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
                        
                        print(f"Creating {conf.data_size} data: "+\
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

                        print(f"{now()}: Evaluating top{k}..")
                        conf_pred = ConformalPrediction(X_cal, y_cal, X_est, y_est, model, conf.delta)
                        
                        print(f"{now()}: Calculating error in test set for top-{k} predictor")
                        p_error_t = conf_pred.error_given_test_set_topk(X_test, y_test, human.w_matrix, k=k)
                        p_error = p_error_t.detach().cpu().numpy()
                        with open(f"{res_dir}/top{k}_test_error", 'wb') as f1:
                            pickle.dump(p_error, f1, pickle.HIGHEST_PROTOCOL)

                        sys.stdout = original_stdout
                    sys.stderr = original_stderr
                except:
                    print(sys.exc_info(), file=f_e)
                    sys.stdout = original_stdout
                    sys.stderr = original_stderr
                    raise


















