from config import conf
import numpy as np
import torch
import torch.nn.functional as F

class ConformalPrediction:
    def __init__(self, X_cal, y_cal, X_est, y_est,model, delta) -> None:
        self.model = model
        self.X_cal = X_cal
        self.y_cal = y_cal
        self.X_est = X_est
        self.y_est = y_est
        self.calibration_size = len(y_cal)
        self.delta = delta
        
    def find_all_alpha_values():  
        pass

    def prediction_sets():
        pass

    def find_a_star():
        pass

class StandardCPgpu(ConformalPrediction):
    # TODO change name of class not to be misleading.
    # It works with or without GPU. 
    # Implements system with both standard and modified conformal prediction.
    """Implementation of the functions of our system"""
    def __init__(self, X_cal, y_cal, X_est, y_est,model, delta) -> None:
        super().__init__(X_cal, y_cal,X_est, y_est, model, delta)
    
    def epsilon_fn(self, delta_n_alphas):
        """Estimation error"""
        delta_n_alphas_t = torch.tensor(delta_n_alphas)    
        epsilon =  torch.sqrt((torch.log(delta_n_alphas_t))/(2*self.calibration_size))
        return epsilon

    def find_all_alpha_values(self):
        """Retruns all 0<alpha<1 values that can be considered"""
        # conformal scores of true labels in calibration set
        model_out = self.model.predict_prob(self.X_cal)
        one_hot = np.eye(conf.n_labels)[self.y_cal]
        true_label_logits = model_out*one_hot
        # needed for the quantiles afterwards
        conf_scores = sorted(1 - true_label_logits[true_label_logits >0 ])  
        self.conf_scores_t = torch.tensor(conf_scores, device=conf.device)
        # finding alpha values
        alphas = 1 - (np.arange(1,self.calibration_size+1) / (self.calibration_size + 1))
        self.alphas = alphas
        self.n_alphas = self.calibration_size
        return alphas

    

    def find_a_star(self, w_matrix, a1_star_idx=None, all_a1_a2=False):
        # TODO change title a_star means ^alpha 
        """Return ^alpha"""
        a_star_idx =-1
        curr_criterion = 0
        alphas1 = self.alphas.flatten()
        qhat_a1 = torch.zeros((1,1), device=conf.device)
        if a1_star_idx is not None: 
            # alphas and quantiles for shifted quantile method given a_1
            quant_a1 =  (np.ceil((1 - alphas1[a1_star_idx])*(self.calibration_size+1))/self.calibration_size)
            qhat_a1 = torch.quantile(self.conf_scores_t, quant_a1)
            alphas = self.alphas[self.alphas > self.alphas[a1_star_idx]]
        else:
            # alphas for standard conformal prediction method
            alphas = self.alphas

        # all quantiles for each alpha value
        quant_unique =  (np.ceil((1 - alphas)*(self.calibration_size+1))/self.calibration_size).flatten()
        # initialize estimation error
        self.epsilon = np.zeros(quant_unique.shape)
        # output scores for each sample in estimation set
        output_scores = 1 - self.model.predict_prob(self.X_est)
        
        # move data to gpu if available
        quants_t = torch.tensor(quant_unique, device=conf.device)
        qhats_t = torch.quantile(self.conf_scores_t, quants_t, keepdim=True)
        qhats_t = qhats_t.unsqueeze(1)
        y_est_t = torch.tensor(self.y_est, device=conf.device, dtype=torch.int64)
        fill_value_t =  torch.tensor(0, dtype=torch.double,device=conf.device)
        output_scores_t =  torch.tensor(output_scores,device=conf.device)
        ws_t = torch.tensor(w_matrix[self.y_est], device=conf.device)

        for i,q in enumerate(qhats_t):
            qhats = q.expand(self.calibration_size,conf.n_labels )
           
            # sets[sample][label] is 1 for the labels in the prediction set for each sample
            if a1_star_idx is not None:
                # sets for shifted quantile method given a_1
                qhats_a1 = qhat_a1.expand(self.calibration_size,conf.n_labels)
                sets_upper = torch.where(output_scores_t <= qhats_a1, 1,0)
                sets_lower = torch.where(qhats <= output_scores_t, 1,0)
                sets = sets_upper* sets_lower
            else:
                # sets for standard conformal prediction method
                sets = torch.where(output_scores_t<= qhats, 1,0)
            sets_exp_ws = sets * torch.exp(ws_t)

             # denominators for all P[\hat Y = Y | C_alpha,  Y \in C_alpha(X), Y=y]
            denominators = torch.sum(sets_exp_ws, axis=1)    
            one_hot_ycal = F.one_hot(y_est_t)
            # mask for prediction sets that include the true label
            mask = sets * one_hot_ycal
            true_label_in_sets_idx = torch.sum(mask, axis=1)
            
            # nominators for all P[\hat Y = Y | C_alpha(X), Y \in C_alpha(X), Y=y]
            nominators = torch.sum(sets_exp_ws*one_hot_ycal, axis=1)

            # apply mask so that Y \in C_alpha(X) is satisfied
            masked_prob = torch.where(true_label_in_sets_idx==1, nominators/denominators, fill_value_t)
            # number of sets that Y \in C_alpha(X) is satisfied
            k_a = true_label_in_sets_idx.sum()        

            if k_a > 0 :
                # Empirical estimation of human expexted success probability when choosing from the prediction sets.
                expected_correct_prob = masked_prob.sum()/self.calibration_size
                delta_n_alphas = (alphas.shape[0] /self.delta) if not all_a1_a2 else (self.calibration_size*(self.calibration_size - 1)/2)/self.delta
                # Estimation error
                epsilon = self.epsilon_fn(delta_n_alphas)
                self.epsilon[i] = epsilon
                #  Compare current alpha with the best alpha so far
                criterion = (expected_correct_prob - epsilon)

                if criterion > curr_criterion:
                    a_star_idx = i
                    curr_criterion = criterion
        if all_a1_a2:
            # Set when searching for a1, a2
            return a_star_idx, curr_criterion

        return a_star_idx


    def error_given_test_set_per_a(self, X_test, y_test, w_matrix, alphas,a_star_idx=None, a2_star_idx=None):
        """Misprediction probability for each value of alpha or alpha_2 given alpha_1"""
        test_size = len(X_test)
        output_scores = 1 - self.model.predict_prob(X_test)
       
       # alphas and quantiles for shifted quantile method
        if a_star_idx is not None: 
            quant_a1=  (np.ceil((1 - self.alphas[a_star_idx])*(self.calibration_size+1))/self.calibration_size)
            qhat_a1 = torch.quantile(self.conf_scores_t, quant_a1)
            alphas = np.array([alphas]) if a2_star_idx is not None else self.alphas[self.alphas > self.alphas[a_star_idx]]

        qhats_unique =  (np.ceil((1 - alphas)*(self.calibration_size+1))/self.calibration_size)
        error_rate_per_a = torch.zeros((len(qhats_unique),), device=conf.device)

        # move data to gpu if available
        qhats_t = torch.tensor(qhats_unique, device=conf.device).unsqueeze(1)
        y_test_t = torch.tensor(y_test, device=conf.device, dtype=torch.int64)
        output_scores_t =  torch.tensor(output_scores,device=conf.device)
        ws_t = torch.tensor(w_matrix[y_test], device=conf.device)
        a_empty_sets = 0
        fill_value_t =  torch.exp(ws_t)/(torch.exp(ws_t).sum(axis=1).unsqueeze(1).expand(-1,conf.n_labels))

        for i,q in enumerate(qhats_t):

            qhats = q.expand(test_size,conf.n_labels )
            # sets[sample][label] is 1 for the labels in the prediction set for each sample
            if a_star_idx is not None:
                # sets for shifted quantile method given alpha_1
                qhats_a1 = qhat_a1.expand(test_size,conf.n_labels )
                sets_upper = torch.where(  output_scores_t <= qhats_a1, 1,0)
                sets_lower = torch.where(qhats <= output_scores_t, 1,0)
                sets = sets_upper* sets_lower
            else:
                sets = torch.where(output_scores_t<=qhats, 1,0)
            non_empty_sets = sets.sum(axis=1).count_nonzero()
            
            if non_empty_sets==0 :
                a_empty_sets+=1

            # Denominators for  P[\hat Y = y | C_alpha(X), y \in C_alpha(X)]
            sets_exp_ws = sets * torch.exp(ws_t)
            denominators_col = torch.sum(sets_exp_ws, axis=1)
            denominators = denominators_col.unsqueeze(1).expand(-1,conf.n_labels)


            # Nomiators for  P[\hat Y = y | C_alpha(X), y \in C_alpha(X)]
            nominators = sets_exp_ws        
        
            # confusion matrix for each prediction set 
            cm = torch.where(denominators>0, nominators/denominators, fill_value_t)

            # human prediction from prediction sets
            y_h = cm.multinomial(num_samples=1, replacement=True, generator=conf.torch_rng).squeeze()

            # Set dummy prediction -1 for empty sets, so that it is counted as misprediction
            y_hats = torch.where(denominators_col>0, y_h , -1)
            # misprediction probability
            errors = (y_hats!=y_test_t).count_nonzero().double()
            error_rate_per_a[i] = errors/test_size
               
                
            
        return error_rate_per_a



    def size_given_test_set_per_a(self, X_test, y_test, w_matrix, alphas,a_star_idx=None, a2_star_idx=None):
        """Average set size for each value of alpha or alpha_2 given alpha_1"""
        test_size = len(X_test)
        output_scores = 1 - self.model.predict_prob(X_test)
    
        if a_star_idx is not None: 
            # alphas and quantiles for shifted quantile method
            quant_a1=  (np.ceil((1 - self.alphas[a_star_idx])*(self.calibration_size+1))/self.calibration_size)
            qhat_a1 = torch.quantile(self.conf_scores_t, quant_a1)
            alphas = np.array([alphas]) if a2_star_idx is not None else self.alphas[self.alphas > self.alphas[a_star_idx]]

        qhats_unique =  (np.ceil((1 - alphas)*(self.calibration_size+1))/self.calibration_size)
        set_size_per_a = torch.zeros((len(qhats_unique),), device=conf.device)

        # move data to gpu if available
        qhats_t = torch.tensor(qhats_unique, device=conf.device).unsqueeze(1)
        output_scores_t =  torch.tensor(output_scores,device=conf.device)
        ws_t = torch.tensor(w_matrix[y_test], device=conf.device)

        for i,q in enumerate(qhats_t):

            qhats = q.expand(test_size,conf.n_labels )
            # sets[sample][label] is 1 for the labels in the prediction set for each sample
            if a_star_idx is not None:
                # sets for shifted quantile method
                qhats_a1 = qhat_a1.expand(test_size,conf.n_labels )
                sets_upper = torch.where(output_scores_t <= qhats_a1,1,0)
                sets_lower = torch.where(qhats <= output_scores_t, 1,0)
                sets = sets_upper* sets_lower
            else:
                sets = torch.where(output_scores_t<=qhats, 1,0)
            size_per_set = sets.sum(axis=1)
            set_size_per_a[i] = size_per_set.sum()/size_per_set.numel()
               
            
        return set_size_per_a



        