from config import conf
from conformal_prediction import StandardCPgpu
import numpy as np
import os
from expert.expert import ExpertSynthetic
import pickle
import utils
from model.model import ModelSynthetic
import sys
import datetime
from tqdm import tqdm


results_root = f"{conf.ROOT_DIR}/results_synthetic"
if not os.path.exists(results_root):
    os.mkdir(results_root)

results_root+=f"/{conf.n_labels}labels_calibrationSet{conf.cal_split}"
if not os.path.exists(results_root):
    os.mkdir(results_root)

for human_accuracy in conf.accuracies:
    for machine_accuracy in conf.accuracies:
        for run in tqdm(range(conf.n_runs_per_split)):

            res_dir = f"{results_root}/human{human_accuracy}_machine{machine_accuracy}_run{run}"
            if not os.path.exists(res_dir):
                os.mkdir(res_dir)
            
            X_train, X_test,X_cal,X_est, y_train, y_test,y_cal, y_est  = utils.make_dataset(run, machine_accuracy)

            conf.accuracy = human_accuracy
            human = ExpertSynthetic(conf)

            model = ModelSynthetic()
            model.train(X_train,y_train)
            print(model.test(X_test, y_test))

            conf_pred = StandardCPgpu(X_cal, y_cal,X_est, y_est, model, conf.delta)
            print(res_dir)
            with open(f"{res_dir}/alphas1",'rb') as f:
                alphas = pickle.load(f)
            
            

            size_t = conf_pred.size_given_test_set_per_a(X_test,y_test, human.w_matrix,alphas)
            size = size_t.detach().cpu().numpy()
            with open(f"{res_dir}/set_size_test",'wb') as f2:
                pickle.dump(size, f2 ,pickle.HIGHEST_PROTOCOL)


            
            print("Done")
