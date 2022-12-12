from config import conf
import numpy as np
import torch
import torch.nn.functional as F

class ConformalPrediction:
    # Implements system with both standard and modified conformal prediction.
    """Implementation and evaluation of the conformal prediction based support system"""
    def __init__(self, X_cal, y_cal, X_est, y_est,model, delta, has_groups=False) -> None:
        self.model = model
        self.X_cal = X_cal
        self.y_cal = y_cal
        self.X_est = X_est
        self.y_est = y_est
        self.calibration_size = len(y_cal)
        self.delta = delta
        # conformal scores of true labels in calibration set
        model_out = self.model.predict_prob(self.X_cal)
        self.has_groups = has_groups
        if not has_groups:
            one_hot = np.eye(conf.n_labels)[self.y_cal]
        else:
            one_hot = np.eye(conf.n_labels)[self.y_cal[:,0]]

        true_label_logits = model_out*one_hot
        conf_scores = sorted(1 - true_label_logits.sum(axis=1))  
        self.conf_scores_t = torch.tensor(conf_scores, device=conf.device)
    
    def epsilon_fn(self, delta_n_alphas, all_a1_a2):
        """Estimation error"""
        delta_n_alphas_t = torch.tensor(delta_n_alphas)    
        n_alphas = self.calibration_size if not all_a1_a2 else (self.calibration_size*(self.calibration_size+1)/2)
        epsilon = torch.sqrt((torch.log(delta_n_alphas_t))/(2*n_alphas))
        return epsilon

    def find_all_alpha_values(self):
        """Returns all 0<alpha<1 values that can be considered given a fixed calibration set"""
        alphas = 1 - (np.arange(1,self.calibration_size + 1) / (self.calibration_size + 1))
        self.alphas = alphas
        self.n_alphas = self.calibration_size
        return alphas
    
    def find_a_star(self, w_matrix, a1_star_idx=None, all_a1_a2=False):
        """Returns the best alpha value or the best alpha_2 value given alpha_1"""
        a_star_idx = -1
        curr_criterion = 0
        # alphas for standard conformal prediction method
        alphas = self.alphas
        if a1_star_idx is not None: 
            # alphas and quantiles for shifted quantile method given a_1
            quant_prob_a1 = np.ceil((1 - alphas[a1_star_idx])*(self.calibration_size+1))/self.calibration_size
            qhat_a1 = torch.zeros((1,1), device=conf.device)
            qhat_a1 = torch.quantile(self.conf_scores_t, quant_prob_a1)
            alphas = self.alphas[self.alphas > self.alphas[a1_star_idx]]
            if all_a1_a2 is not None:
                alphas = np.append(alphas, 1)

        # quantile probabilities for each alpha value
        quant_prob = (np.ceil((1 - alphas)*(self.calibration_size+1))/self.calibration_size).flatten()
        
        # output scores for each sample in estimation set
        output_scores = 1 - self.model.predict_prob(self.X_est)
        
        # move data to gpu if available
        quant_probs_t = torch.tensor(quant_prob, device=conf.device)
        qhats_t = torch.quantile(self.conf_scores_t, quant_probs_t, keepdim=True)
        qhats_t = qhats_t.unsqueeze(1)
        y_est_t = torch.tensor(self.y_est, device=conf.device, dtype=torch.int64)
        fill_value_t = torch.tensor(0, dtype=torch.double, device=conf.device)
        output_scores_t = torch.tensor(output_scores, device=conf.device)
        if not self.has_groups:
            ws_t = torch.tensor(w_matrix[self.y_est], device=conf.device)
        else:
            ws_t = torch.tensor(w_matrix[self.y_est[:,1],self.y_est[:,0]], device=conf.device)
        
        # estimation error
        delta_n_alphas = (alphas.shape[0]/self.delta) if not all_a1_a2 else (self.calibration_size*(self.calibration_size + 1)/2)/self.delta
        epsilon = self.epsilon_fn(delta_n_alphas, all_a1_a2)

        for i,q in enumerate(qhats_t):
            qhats = q.expand(self.calibration_size, conf.n_labels)
           
            # sets[sample][label] is 1 for the labels in the prediction set for each sample
            if a1_star_idx is not None:
                # sets for shifted quantile method given a_1
                qhats_a1 = qhat_a1.expand(self.calibration_size, conf.n_labels)
                sets_upper = torch.where(output_scores_t <= qhats_a1, 1, 0)
                sets_lower = torch.where(qhats < output_scores_t, 1, 0)
                sets = sets_upper* sets_lower
            else:
                # sets for standard conformal prediction method
                sets = torch.where(output_scores_t <= qhats, 1, 0)
            sets_exp_ws = sets * torch.exp(ws_t)

            # denominators for all P[\hat Y = Y ; C_alpha |  Y \in C_alpha(X), Y=y]
            denominators = torch.sum(sets_exp_ws, axis=1)    
            if not self.has_groups:
                one_hot_ycal = F.one_hot(y_est_t)
            else:
                one_hot_ycal = F.one_hot(y_est_t[:,0])

            # mask for prediction sets that include the true label
            mask = sets * one_hot_ycal
            true_label_in_sets_idx = torch.sum(mask, axis=1)
            
            # nominators for all P[\hat Y = Y ; C_alpha | Y \in C_alpha(X), Y=y]
            nominators = torch.sum(sets_exp_ws*one_hot_ycal, axis=1)

            # apply mask so that Y \in C_alpha(X) is satisfied
            masked_prob = torch.where(true_label_in_sets_idx==1, nominators/denominators, fill_value_t)
            
            # empirical estimation of human expected success probability when choosing from the prediction sets.
            expected_correct_prob = masked_prob.sum()/self.calibration_size
           
            criterion = (expected_correct_prob - epsilon)

            if criterion > curr_criterion:
                a_star_idx = i
                curr_criterion = criterion
        if all_a1_a2:
            # set all_a1_a2=True when searching for the best a1, a2
            return a_star_idx, curr_criterion, criterion

        return a_star_idx

    def error_given_test_set_given_a(self, X_test, y_test, w_matrix, alpha1_value, alpha2_value=None):
        """Empirical expert misprediction probability given the value of alpha or the values alpha_1, alpha_2 during test"""
        test_size = len(X_test)
        output_scores = 1 - self.model.predict_prob(X_test)
       
       # alphas and quantiles for shifted quantile method
        if alpha2_value is not None: 
            quant_prob_a2 = (np.ceil((1 - alpha2_value)*(self.calibration_size+1))/self.calibration_size)
            qhat_a2 = torch.quantile(self.conf_scores_t, quant_prob_a2)

        quant_prob = (np.ceil((1 - alpha1_value)*(self.calibration_size+1))/self.calibration_size)
         
        # move data to gpu if available
        quanta1_prob_t = torch.tensor(quant_prob, device=conf.device)
        qhata1_t = torch.quantile(self.conf_scores_t, quanta1_prob_t, keepdim=True).unsqueeze(1)
        y_test_t = torch.tensor(y_test, device=conf.device, dtype=torch.int64)
        output_scores_t = torch.tensor(output_scores, device=conf.device)
        if not self.has_groups:
            ws_t = torch.tensor(w_matrix[y_test], device=conf.device)
        else:
            ws_t = torch.tensor(w_matrix[y_test[:,1], y_test[:,0]], device=conf.device)
        
        fill_value_t = torch.exp(ws_t)/(torch.exp(ws_t).sum(axis=1).unsqueeze(1).expand(-1,conf.n_labels))

        qhats_a1 = qhata1_t.expand(test_size, conf.n_labels)
        # sets[sample][label] is 1 for the labels in the prediction set for each sample
        sets = torch.where(output_scores_t <= qhats_a1, 1, 0)
        if alpha2_value is not None:
            # sets for shifted quantile method given alpha_1
            qhats_a2 = torch.ones((test_size,conf.n_labels), device=conf.device)*qhat_a2
            sets_lower = torch.where(qhats_a2 < output_scores_t, 1, 0)
            sets = sets * sets_lower
        
        # denominators for  P[\hat Y = y ; C_alpha | y \in C_alpha(X)]
        sets_exp_ws = sets * torch.exp(ws_t)
        denominators_col = torch.sum(sets_exp_ws, axis=1)
        denominators = denominators_col.unsqueeze(1).expand(-1, conf.n_labels)

        # nominators for P[\hat Y = y ; C_alpha| y \in C_alpha(X)]
        nominators = sets_exp_ws        
    
        # confusion matrix for each prediction set 
        cm = torch.where(denominators>0, nominators/denominators, fill_value_t)

        # human predictions from prediction sets
        y_h = cm.multinomial(num_samples=1, replacement=True, generator=conf.torch_rng).squeeze()

        # set dummy prediction -1 for empty sets, so that it is counted as misprediction
        y_hats = torch.where(denominators_col>0, y_h , -1)
       
        # misprediction probability
        if not self.has_groups:
            errors = (y_hats!=y_test_t).count_nonzero().double()
        else:
            errors = (y_hats!=y_test_t[:,0]).count_nonzero().double()
            
        return errors/test_size

    def error_given_test_set_per_a(self, X_test, y_test, w_matrix, alphas, a_star_idx=None):
        """Empirical expert misprediction probability for each value of alpha or alpha_2 given alpha_1 during test"""
        test_size = len(X_test)
        output_scores = 1 - self.model.predict_prob(X_test)
       
       # alphas and quantiles for shifted quantile method
        if a_star_idx is not None: 
            quant_a1 = (np.ceil((1 - self.alphas[a_star_idx])*(self.calibration_size+1))/self.calibration_size)
            qhat_a1 = torch.quantile(self.conf_scores_t, quant_a1)
            alphas = self.alphas[self.alphas > self.alphas[a_star_idx]]

        quant_prob = (np.ceil((1 - alphas)*(self.calibration_size+1))/self.calibration_size)
         
        error_rate_per_a = torch.zeros((len(quant_prob),), device=conf.device)

        # move data to gpu if available
        quant_prob_t = torch.tensor(quant_prob, device=conf.device)
        qhats_t = torch.quantile(self.conf_scores_t, quant_prob_t, keepdim=True).unsqueeze(1)
        y_test_t = torch.tensor(y_test, device=conf.device, dtype=torch.int64)
        output_scores_t = torch.tensor(output_scores, device=conf.device)
        if not self.has_groups:
            ws_t = torch.tensor(w_matrix[y_test], device=conf.device)
        else:
            ws_t = torch.tensor(w_matrix[y_test[:,1], y_test[:,0]], device=conf.device)
 
        a_empty_sets = 0
        fill_value_t = torch.exp(ws_t)/(torch.exp(ws_t).sum(axis=1).unsqueeze(1).expand(-1, conf.n_labels))

        for i,q in enumerate(qhats_t):

            qhats = q.expand(test_size, conf.n_labels)
            # sets[sample][label] is 1 for the labels in the prediction set for each sample
            if a_star_idx is not None:
                # sets for shifted quantile method given alpha_1
                qhats_a1 = qhat_a1.expand(test_size,conf.n_labels )
                sets_upper = torch.where(output_scores_t <= qhats_a1 ,1 ,0)
                sets_lower = torch.where(qhats < output_scores_t, 1, 0)
                sets = sets_upper * sets_lower
            else:
                sets = torch.where(output_scores_t <= qhats, 1, 0)
            non_empty_sets = sets.sum(axis=1).count_nonzero()
            
            if non_empty_sets==0 :
                a_empty_sets+=1

            # denominators for  P[\hat Y = y ; C_alpha | y \in C_alpha(X)]
            sets_exp_ws = sets * torch.exp(ws_t)
            denominators_col = torch.sum(sets_exp_ws, axis=1)
            denominators = denominators_col.unsqueeze(1).expand(-1, conf.n_labels)

            # nominators for P[\hat Y = y ; C_alpha | y \in C_alpha(X)]
            nominators = sets_exp_ws        
        
            # confusion matrix for each prediction set 
            cm = torch.where(denominators>0, nominators/denominators, fill_value_t)

            # human predictions from prediction sets
            y_h = cm.multinomial(num_samples=1, replacement=True, generator=conf.torch_rng).squeeze()

            # set dummy prediction -1 for empty sets, so that it is counted as misprediction
            y_hats = torch.where(denominators_col>0, y_h , -1)
            # misprediction probability
            if not self.has_groups:
                errors = (y_hats!=y_test_t).count_nonzero().double()
            else:
                errors = (y_hats!=y_test_t[:,0]).count_nonzero().double()

            error_rate_per_a[i] = errors/test_size
               
        return error_rate_per_a

    def test_error_robustness(self, p, X_test, y_test, w_matrix, alpha1_value, alpha2_value=None):
        """Empirical expert misprediction probability during test under IIA violations"""
        test_size = len(X_test)
        output_scores = 1 - self.model.predict_prob(X_test)
       
       # alphas and quantiles for shifted quantile method
        if alpha2_value is not None: 
            quant_prob_a2 = (np.ceil((1 - alpha2_value)*(self.calibration_size+1))/self.calibration_size)
            qhat_a2 = torch.quantile(self.conf_scores_t, quant_prob_a2)

        quant_prob = (np.ceil((1 - alpha1_value)*(self.calibration_size+1))/self.calibration_size)

        # move data to gpu if available
        quanta1_prob_t = torch.tensor(quant_prob, device=conf.device)
        qhata1_t = torch.quantile(self.conf_scores_t, quanta1_prob_t, keepdim=True).unsqueeze(1)
        y_test_t = torch.tensor(y_test, device=conf.device, dtype=torch.int64)
        output_scores_t = torch.tensor(output_scores, device=conf.device)
        ws_t = torch.tensor(w_matrix[y_test], device=conf.device)
        fill_value_t = torch.exp(ws_t)/(torch.exp(ws_t).sum(axis=1).unsqueeze(1).expand(-1, conf.n_labels))
        qhats_a1 = qhata1_t.expand(test_size, conf.n_labels)
        
        # sets[sample][label] is 1 for the labels in the prediction set for each sample
        sets = torch.where(output_scores_t <= qhats_a1, 1, 0)
        if alpha2_value is not None:
            # sets for shifted quantile method given alpha_1
            qhats_a2 = torch.ones((test_size,conf.n_labels), device=conf.device)*qhat_a2
            sets_lower = torch.where(qhats_a2 < output_scores_t, 1, 0)
            sets = sets * sets_lower
        
        # denominators for  P[\hat Y = y ; C_alpha | y \in C_alpha(X)]
        sets_exp_ws = sets * torch.exp(ws_t)

        # nominators for P[\hat Y = y ; C_alpha | y \in C_alpha(X)]
        nominators = sets_exp_ws        

        # sets that include the true label
        one_hot_ycal = F.one_hot(y_test_t)
        mask_sets_with_true_labels = (sets * one_hot_ycal).sum(axis=1, keepdim=True).expand(test_size, conf.n_labels)

        # labels excluded from the sets
        mass_of_labels_not_in_the_set = p * ((1 - sets)*torch.exp(ws_t)).sum(axis=1, keepdim=True).expand(test_size, conf.n_labels)

        # sizes of prediction sets - 1
        sets_sizes_minus1 = sets.sum(axis=1, keepdim=True).expand(test_size, conf.n_labels) - 1
        
        mass_to_add_in_false_labels = torch.where( (one_hot_ycal==0) & (sets==1) & (mask_sets_with_true_labels==1), mass_of_labels_not_in_the_set/sets_sizes_minus1, 0)
        
        # tweaked nominators
        tweaked_nominators = nominators+mass_to_add_in_false_labels 
        denominators_col = tweaked_nominators.sum(axis=1)
        denominators = tweaked_nominators.sum(axis=1, keepdim=True).expand(test_size, conf.n_labels)
        tweaked_cm = torch.where(denominators>0, tweaked_nominators/denominators, fill_value_t)

        # human predictions from prediction sets
        y_h = tweaked_cm.multinomial(num_samples=1, replacement=True, generator=conf.torch_rng).squeeze()

        # set dummy prediction -1 for empty sets, so that it is counted as misprediction
        y_hats = torch.where(denominators_col>0, y_h , -1)
       
        # misprediction probability
        errors = (y_hats!=y_test_t).count_nonzero().double()
            
        return errors/test_size

    def size_given_test_set_per_a(self, X_test, alphas, a_star_idx=None):
        """Empirical average set size for each value of alpha or alpha_2 given alpha_1"""
        test_size = len(X_test)
        output_scores = 1 - self.model.predict_prob(X_test)
        
        if a_star_idx is not None: 
            # alphas and quantiles for shifted quantile method
            quant_a1 = (np.ceil((1 - alphas[a_star_idx])*(self.calibration_size+1))/self.calibration_size)
            qhat_a1 = torch.quantile(self.conf_scores_t, quant_a1)
            alphas =  alphas[alphas > alphas[a_star_idx]]

        quant_prob_t = torch.tensor(np.ceil((1 - alphas)*(self.calibration_size+1))/self.calibration_size, device=conf.device)
        set_size_per_a = torch.zeros((len(alphas),), device=conf.device)

        # move data to gpu if available
        qhats_t = torch.quantile(self.conf_scores_t, quant_prob_t, keepdim=True).unsqueeze(1)
        output_scores_t = torch.tensor(output_scores, device=conf.device)

        for i,q in enumerate(qhats_t):

            qhats = q.expand(test_size, conf.n_labels)
            # sets[sample][label] is 1 for the labels in the prediction set for each sample
            if a_star_idx is not None:
                # sets for shifted quantile method
                qhats_a1 = qhat_a1.expand(test_size, conf.n_labels)
                sets_upper = torch.where(output_scores_t <= qhats_a1, 1, 0)
                sets_lower = torch.where(qhats < output_scores_t, 1, 0)
                sets = sets_upper * sets_lower
            else:
                sets = torch.where(output_scores_t <= qhats, 1, 0)
            size_per_set = sets.sum(axis=1)
            set_size_per_a[i] = size_per_set.sum()/size_per_set.numel()
               
        return set_size_per_a    

    def error_given_test_set_topk(self, X_test, y_test, w_matrix, k=5):
        """Emprical misprediction probability of an expert using a top-k predictor"""
        test_size = len(X_test)
        output_scores = self.model.predict_prob(X_test)
        error_rate_per_a = torch.zeros((1,), device=conf.device)

        # move data to gpu if available
        y_test_t = torch.tensor(y_test, device=conf.device, dtype=torch.int64)
        output_scores_t = torch.tensor(output_scores, device=conf.device)
        ws_t = torch.tensor(w_matrix[y_test], device=conf.device)
        fill_value_t = torch.exp(ws_t)/(torch.exp(ws_t).sum(axis=1).unsqueeze(1).expand(-1, conf.n_labels))
        
        # compute topk labels for each sample
        sorted_output_scores = torch.topk(output_scores_t, k=k, dim=1).indices
        
        # prediction sets with topk labels
        sets = torch.any(F.one_hot(sorted_output_scores, num_classes=conf.n_labels), 1).int()

        # denominators for  P[\hat Y = y ; C_k | y \in C_k(X)]
        sets_exp_ws = sets * torch.exp(ws_t)
        denominators_col = torch.sum(sets_exp_ws, axis=1)
        denominators = denominators_col.unsqueeze(1).expand(-1, conf.n_labels)

        # nominators for P[\hat Y = y ; C_k | y \in C_k(X)]
        nominators = sets_exp_ws        
    
        # confusion matrix for each prediction set 
        cm = torch.where(denominators>0, nominators/denominators, fill_value_t)

        # human predictions from prediction sets
        y_h = cm.multinomial(num_samples=1, replacement=True, generator=conf.torch_rng).squeeze()

        # set dummy prediction -1 for empty sets, so that it is counted as misprediction
        y_hats = torch.where(denominators_col>0, y_h , -1)
        
        # misprediction probability
        errors = (y_hats!=y_test_t).count_nonzero().double()
        error_rate_per_a = errors/test_size
            
        return error_rate_per_a

    def size_given_test_set_given_a(self, X_test, alpha, alpha2=None):
        """Set size distribution for given alpha or alpha_1, alpha_2 during test"""
        test_size = len(X_test)
        output_scores = 1 - self.model.predict_prob(X_test)
    
        if alpha2 is not None: 
            # alphas and quantiles for shifted quantile method
            quant_a1 = (np.ceil((1 - alpha)*(self.calibration_size+1))/self.calibration_size)
            qhat_a1 = torch.quantile(self.conf_scores_t, quant_a1)
            alpha = alpha2

        quant_prob_t = torch.tensor(np.ceil((1 - alpha)*(self.calibration_size+1))/self.calibration_size, device=conf.device)

        # move data to gpu if available
        qhat_t = torch.quantile(self.conf_scores_t, quant_prob_t)
        output_scores_t = torch.tensor(output_scores,device=conf.device)
        qhats = qhat_t.expand(test_size, conf.n_labels)

        # sets[sample][label] is 1 for the labels in the prediction set for each sample
        if alpha2 is not None:
            # sets for shifted quantile method
            qhats_a1 = qhat_a1.expand(test_size, conf.n_labels)
            sets_upper = torch.where(output_scores_t <= qhats_a1, 1, 0)
            sets_lower = torch.where(qhats < output_scores_t, 1, 0)
            sets = sets_upper * sets_lower
        else:
            sets = torch.where(output_scores_t <= qhats, 1, 0)
        
        size_per_set = sets.sum(axis=1)
        set_sizes_t, counts_t = torch.unique(size_per_set, return_counts=True)
        
        return set_sizes_t, counts_t

    def empirical_coverage(self, X_test, y_test, alpha1, alpha2=None):
        """Empirical coverage on test set given alpha or alpha1, alpha2"""
        test_size = len(X_test)
        output_scores = 1 - self.model.predict_prob(X_test)
       
       # quantiles for shifted quantile method
        if alpha2 is not None: 
            quant_a2 = (np.ceil((1 - alpha2)*(self.calibration_size+1))/self.calibration_size)
            qhat_a2 = torch.quantile(self.conf_scores_t, quant_a2)

        quant_prob_t = torch.tensor(np.ceil((1 - alpha1)*(self.calibration_size+1))/self.calibration_size, device=conf.device)

        # move data to gpu if available
        qhat = torch.quantile(self.conf_scores_t, quant_prob_t) 
        y_test_t = torch.tensor(y_test, device=conf.device, dtype=torch.int64)
        output_scores_t = torch.tensor(output_scores, device=conf.device)
        qhats = torch.ones((test_size,conf.n_labels), device=conf.device)*qhat

        # sets[sample][label] is 1 for the labels in the prediction set for each sample
        sets = torch.where(output_scores_t <= qhats, 1, 0)
        if alpha2 is not None:
            # sets for shifted quantile method given alpha_1
            qhats_a2 = torch.ones((test_size,conf.n_labels), device=conf.device)*qhat_a2
            sets_lower = torch.where(qhats_a2 < output_scores_t, 1, 0)
            sets = sets * sets_lower
        
        one_hot_ycal = F.one_hot(y_test_t)
        # mask for prediction sets that include the true label
        true_label_in_sets = sets * one_hot_ycal
        return true_label_in_sets.sum()/test_size    
