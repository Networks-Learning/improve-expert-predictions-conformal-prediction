import numpy as np
import pandas as pd
import seaborn as sns
from config import conf
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import utils
from expert.expert import ExpertReal
import torch 

legend_font_size = 32
marker_size_main = 250
marker_size_hat = 200
markerscale = 4

def plot_a(alphas, errors, alpha_star_idx, alpha_star_star_idx=None):
    """ Plot alpha vs error with ^alpha  and alpha* highlighted"""
    alphas_flat = alphas
    alphas_flat_sorted = np.unique(sorted(alphas_flat))

    alpha_star = alphas_flat[alpha_star_idx]
    marker_points = [np.argwhere(alphas_flat_sorted== alpha_star)[0][0]]
    if alpha_star_star_idx is not None:
        alpha_star_star = alphas_flat[alpha_star_star_idx]
        marker_points.append(np.argwhere(alphas_flat_sorted== alpha_star_star)[0][0])


    df = pd.DataFrame(list(zip(alphas, errors)), columns=["alpha", "error"])

    sns.lineplot(data=df, x="alpha", y="error",marker="o",markersize=10, markevery=marker_points)

def read_data(dir):
    """Read results"""
    with open(f"{dir}/alphas1","rb") as f:
        alphas1 = pickle.load(f)

    with open(f"{dir}/alpha_1","rb") as f:
        alpha_1 = pickle.load(f)

    with open(f"{dir}/alpha_2","rb") as f:
        alpha_2 = pickle.load(f)
    
    try:
        with open(f"{dir}/set_size_test",'rb') as f:
            size = pickle.load(f)
    except:
        size = 0
        pass

    with open(f"{dir}/alpha1_test_error","rb") as f:
        perror1 = pickle.load(f)

    with open(f"{dir}/alpha2_test_error","rb") as f:
        perror2 = pickle.load(f)

    return alphas1, alpha_1, alpha_2, perror1, perror2, size


def print_accuracy_synthetic(split=.15):
    """Information of Table 1"""
    entries = []
    best = []
    std_error = []
    for i, labels in enumerate([10, 50, 100]):
        for j,split in enumerate([split]):
            entries = []
            base  = f"results_synthetic/{labels}labels_calibrationSet{split}"
            std_error = []
            for human_accuracy in conf.accuracies:
                for machine_accuracy in conf.accuracies:
                    best = []
                    for run in range(5):
                        dir = f"{base}\human{human_accuracy}_machine{machine_accuracy}_run{run}"
                        
                        alphas1, alpha_1, alpha_2, perror1, perror2,_ = read_data(dir)
                        best.append(np.maximum(1 - perror1[alpha_1], 1 - perror2[alpha_2])) 
                
                    entries.append((human_accuracy, machine_accuracy, np.mean(best)))
                    std_error.append((human_accuracy, machine_accuracy,  np.std(best)/np.sqrt(len(best))))


            df = pd.DataFrame(entries, columns=["Human", "Machine", "Accuracy" ])
            
            pvt_df = df.pivot(index='Human', columns="Machine", values="Accuracy")


            df_std = pd.DataFrame(std_error, columns=["Human", "Machine", "Variance" ])

            pvt_df_std = df_std.pivot(index='Human', columns="Machine", values="Variance")
            # annot_std = (pvt_df_std).round(2).astype("string") # std error

            annot = (pvt_df).round(2).astype("string") 

            print(annot.to_latex())
            # print(annot_std.to_latex())



def plot_alpha_four_confs(method=0,labels=10, acc1=0.3, acc2=0.7, split=0.15, run=0):
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amsfonts, geometry}'
    mpl.rcParams['axes.formatter.use_mathtext'] = True
    plt.rcParams.update({
    "text.usetex": True,
    "font.serif": ["Computer Modern Roman"],
    "font.size": 38,
    "figure.figsize":(15,9)
    })

    acc_a_to_df = []
    marker_x = []
    marker_y = []
    human_str = r'$\mathbb{P}[\hat Y = Y \,;\, \mathcal{Y}] = '
    machine_str = r'\mathbb{P}[Y^\prime = Y] = '
    for human_accuracy,machine_accuracy in [(acc1,acc2),(acc2,acc2),(acc2,acc1), (acc1,acc1)]:
        base = f"results_synthetic/{labels}labels_calibrationSet{split}"
        dir = f"{base}\human{human_accuracy}_machine{machine_accuracy}_run{run}"
        

        alphas1, alpha_1, alpha_2, perror1, perror2 ,_ = read_data(dir)
        
        # method 2
        if method:
            alphas1 = alphas1[alphas1 > alphas1[alpha_1]]
            perror1 = perror2
            alpha_1 = alpha_2
        
        # markers for ^alpha
        alpha_star = alphas1[alpha_1]
        marker_x.append(alpha_star)
        marker_y.append(1 - perror1[alpha_1])
        
        for a, err in list(zip(alphas1, perror1)):
            acc_a_to_df.append((f"{human_str}{human_accuracy}, {machine_str}{machine_accuracy}$", a, 1 - err))

    df = pd.DataFrame(acc_a_to_df, columns=["human_machine", r'$\alpha$', r'$\mathbb{P}[\hat Y =Y \,;\, \mathcal{C}_{\alpha}]$'])
    
   
    markers = {f"{human_str}{acc1}, {machine_str}{acc2}$" : 'X',
                f"{human_str}{acc2}, {machine_str}{acc2}$": 'X',
                f"{human_str}{acc2}, {machine_str}{acc1}$": 'o',
                f"{human_str}{acc1}, {machine_str}{acc1}$": 'o'}
    ax = sns.scatterplot(data=df, x=r'$\alpha$', y=r'$\mathbb{P}[\hat Y =Y \,;\, \mathcal{C}_{\alpha}]$'\
                , hue="human_machine", style="human_machine" ,markers=markers , edgecolor=None, palette='colorblind', s=marker_size_main, rasterized=True)
    ax.yaxis.labelpad = 20
    ax.legend(fontsize=legend_font_size, loc='lower left', markerscale=markerscale)
    
    c = ['darkblue','saddlebrown','darkgreen','darkred']
    ax.scatter(x=marker_x, y=marker_y, color=c, marker='s', s=marker_size_hat)

    plt.savefig(f"{conf.ROOT_DIR}/{acc1}_{acc2}_alpha_sythetic_m{method}.pdf")

def plot_size_alpha_four_confs(method=0,labels=10, acc1=0.5, acc2=0.9, split=0.15, run=0):
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amsfonts, geometry}'
    mpl.rcParams['axes.formatter.use_mathtext'] = True
    plt.rcParams.update({
    "text.usetex": True,
    "font.serif": ["Computer Modern Roman"],
    "font.size": 38,
    "figure.figsize":(15,9)
    })

    size_a_to_df = []
    marker_x = []
    marker_y = []
    human_str = r'$\mathbb{P}[\hat Y = Y \,;\, \mathcal{Y}] = '
    machine_str = r'$\mathbb{P}[Y^\prime = Y] = '
    human_accuracy = 0.5
    for machine_accuracy in [acc1,acc2]:
        for human_accuracy in [acc1,acc2]:
            base = f"results_synthetic/{labels}labels_calibrationSet{split}"
            run = 0
            dir = f"{base}\human{human_accuracy}_machine{machine_accuracy}_run{run}"

            alphas1, alpha_1, alpha_2, perror1, perror2 ,size = read_data(dir)

            alpha_star = alphas1[alpha_1]
            marker_x.append(alpha_star)
            marker_y.append(size[alpha_1])
        
        for a, s in list(zip(alphas1, size)):
            size_a_to_df.append((f"{machine_str}{machine_accuracy}$", a, s))
    exp_size_str = r'$\mathbb{E}[|\mathcal{C}_{\alpha}(X)|]$'
    df = pd.DataFrame(size_a_to_df, columns=["human_machine",  r'$\alpha$', exp_size_str ])
    
    palette = {f"{machine_str}{acc1}$": 'lightgray',f"{machine_str}{acc2}$": 'darkgray'}

    ax = sns.scatterplot(data=df, x=r'$\alpha$', y=exp_size_str\
                , hue="human_machine", hue_order=[f"{machine_str}{acc2}$", f"{machine_str}{acc1}$"], edgecolor=None,style="human_machine" ,palette=palette,  s=marker_size_main, rasterized=True)
    ax.yaxis.labelpad = 20
    ax.legend(fontsize=legend_font_size, loc='upper right', markerscale=markerscale)

    c = ['darkred','darkgreen', 'darkblue','saddlebrown']
    ax.scatter(x=marker_x, y=marker_y,c=c, marker='s', s=marker_size_hat)
    

    plt.savefig(f"{conf.ROOT_DIR}/{acc1}_{acc2}_size_alpha_sythetic_m{method}.pdf")



def get_mn():
    """Relative gain in success probability for 
       all splits and number of labels in synthetic experiments """
    entries = []
    n_cnt = []
    for split in [0.02,0.05,0.1,0.15]:
        for i, labels in enumerate([10, 50, 100]):
            base = f"results_synthetic/{labels}labels_calibrationSet{split}"
            n_el = 0
            for human_accuracy in conf.accuracies:
                for machine_accuracy in conf.accuracies:
                    for run in range(5):
                        dir = f"{base}\human{human_accuracy}_machine{machine_accuracy}_run{run}"
                        
                        alphas1, alpha_1, alpha_2, perror1, perror2, _ = read_data(dir)
                        gain = (1 - perror1[alpha_1]- human_accuracy )/ human_accuracy 
                        if gain > 0:
                            n_data = int((conf.data_size* (1 - conf.test_split))*split)
                            entries.append((n_data, labels, gain*100))
                            n_el+=1
            n_cnt.append(n_el)
    succ_p = r'$\textsc{Success probability gain  }\%$'
    df = pd.DataFrame(entries, columns=[r'$m$', r'$n$', succ_p ])
    

    mean_df_m = df.groupby([ r'$m$', r'$n$']).mean()
    std_df_m = df.groupby([ r'$m$', r'$n$']).std()[succ_p]/np.sqrt(np.asarray(n_cnt).T)

    mean_df_n = df.groupby([ r'$n$', r'$m$']).mean()
    std_df_n = df.groupby([ r'$n$', r'$m$']).std()[succ_p]/np.sqrt(np.asarray(n_cnt).T)
   

    return mean_df_m, std_df_m, mean_df_n, std_df_n


def print_accuracy_tables_real(split):
    "Information of Table 2"
    entries = []
    best = []
    std_error = []
 
    values = [r'$\textsc{DenseNet-BC}$', r'$\textsc{PreResNet-110}$',r'$\textsc{ResNet-110}']
    model_name_mapping = dict(zip( conf.model_names, values))
    for j,split in enumerate([split]):
        entries = []
        base  = f"results_real/calibrationSet{split}"
        std_error = []
        for machine_model in conf.model_names:
            best = []
            for run in range(10):
                dir = f"{base}\{machine_model}_run{run}"
                
                alphas1, alpha_1, alpha_2, perror1, perror2,_ = read_data(dir)
                best.append(1 - perror1[alpha_1])
        
            entries.append((model_name_mapping[machine_model], np.mean(best)))
            std_error.append((model_name_mapping[machine_model],  np.std(best)/np.sqrt(len(best))))


        df = pd.DataFrame(entries, columns=["Machine", "Accuracy" ])
        
        pvt_df = df.pivot(columns='Machine', values="Accuracy")


        df_std = pd.DataFrame(std_error, columns=[ "Machine", "Variance" ])

        pvt_df_std = df_std.pivot(columns="Machine", values="Variance")

        annot = pvt_df.round(3).astype("string") + "$\pm$" + (pvt_df_std).round(3).astype("string")

        print(annot.to_latex())


def plot_alpha_real( split=0.15, run=0):
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amsfonts, geometry}'
    mpl.rcParams['axes.formatter.use_mathtext'] = True
    plt.rcParams.update({
    "text.usetex": True,
    "font.serif": ["Computer Modern Roman"],
    "font.size": 38,
    "figure.figsize":(15,9)
    })
    acc_a_to_df = []
    marker_points = []
    values = [r'$\textsc{DenseNet-BC}$', r'$\textsc{PreResNet-110}$',r'$\textsc{ResNet-110}$']
    model_name_mapping = dict(zip( conf.model_names, values))
    machine_str = r'\mathbb{P}[Y^\prime = Y] = '
    base = f"{conf.ROOT_DIR}/results_real/calibrationSet{split}"
    marker_x = []
    marker_y = []
    
    for machine_model in conf.model_names:
        dir = f"{base}\{machine_model}_run{run}"
     
        alphas1, alpha_1, alpha_2, perror1, perror2 , _ = read_data(dir)
               
        alphas_flat_sorted = np.unique(sorted(alphas1))
        
        alpha_star = alphas1[alpha_1]
        marker_points.append(np.argwhere(alphas_flat_sorted== alpha_star)[0][0])
        marker_x.append(alpha_star)
        marker_y.append(1 - perror1[alpha_1])
        
        for a, err in list(zip(alphas1, perror1)):
            acc_a_to_df.append((f"{model_name_mapping[machine_model]}", a, 1 - err))

    df = pd.DataFrame(acc_a_to_df, columns=["model", r'$\alpha$', r'$\mathbb{P}[\hat Y =Y \,;\, \mathcal{C}_{\alpha}]$'])

    ax = sns.scatterplot(data=df, x=r'$\alpha$', y=r'$\mathbb{P}[\hat Y =Y \,;\, \mathcal{C}_{\alpha}]$'\
                , hue="model", style='model', palette='colorblind', s=marker_size_main , edgecolor=None,rasterized=True)
    ax.yaxis.labelpad = 20
    ax.legend(fontsize=legend_font_size, loc='lower left',  markerscale=markerscale)
    
    ax.scatter(x=marker_x, y=marker_y, color=["darkblue","saddlebrown","darkgreen"], marker='s', s=marker_size_hat)
    
    
    ax.set(xscale='log')
  

    plt.savefig(f"{conf.ROOT_DIR}/real_alpha.pdf")



def plot_size_alpha_real( split=0.15, run=0):
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amsfonts, geometry}'
    mpl.rcParams['axes.formatter.use_mathtext'] = True
    plt.rcParams.update({
    "text.usetex": True,
    # "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 38,
    "figure.figsize":(15,9)
    })
    acc_a_to_df = []
    marker_points = []
    values = [r'$\textsc{DenseNet-BC}$', r'$\textsc{PreResNet-110}$',r'$\textsc{ResNet-110}$']
    model_name_mapping = dict(zip( conf.model_names, values))
    machine_str = r'\mathbb{P}[Y^\prime = Y] = '
    base = f"{conf.ROOT_DIR}/results_real/calibrationSet{split}"
    marker_x = []
    marker_y = []
   
    for machine_model in conf.model_names:
        run = 0    
        dir = f"{base}\{machine_model}_run{run}"
            
        alphas1, alpha_1, alpha_2, perror1, perror2 , size = read_data(dir)
        alpha_star = alphas1[alpha_1]
        marker_x.append(alpha_star)
        marker_y.append(size[alpha_1])

        for a, s in list(zip(alphas1, size)):
            acc_a_to_df.append((f"{model_name_mapping[machine_model]}", run, a, s))
    
    exp_size_str = r'$\mathbb{E}[|\mathcal{C}_{\alpha}(X)|]$'
    df = pd.DataFrame(acc_a_to_df, columns=["model",'run', r'$\alpha$', exp_size_str])

    ax = sns.scatterplot(data=df, x=r'$\alpha$', y=exp_size_str\
                , hue="model", style='model', palette='colorblind', s=marker_size_main ,edgecolor=None, rasterized=True)

   
    
    ax.yaxis.labelpad = 20
    ax.legend(fontsize=legend_font_size, loc='upper right', markerscale=markerscale)
    ax.scatter(x=marker_x, y=marker_y,c=["darkblue","saddlebrown","darkgreen"], marker='s', s=marker_size_hat)

    
    ax.set(yscale='log')
    ax.set(xscale='log')
    

    plt.savefig(f"{conf.ROOT_DIR}/real_size_alpha.pdf")


def get_m_real():
    """Relative gain in success probability for all splits in real data experiments"""
    entries = []
    for split in [0.02, 0.05, 0.1,0.15]:
        base = f"results_real/calibrationSet{split}"
        for model in conf.model_names:
            for run in range(10):
                dir = f"{base}\{model}_run{run}"
                human_accuracy = 0.95
                with open(f"{dir}/logs.txt", "r") as f:
                    lines = f.readlines()
                    m_tmp = float(lines[3])
                    machine_accuracy = np.round(m_tmp, 3)
                alphas1, alpha_1, alpha_2, perror1, perror2,_ = read_data(dir)
                gain = (1 - perror1[alpha_1] - np.maximum(human_accuracy, machine_accuracy) )/ np.maximum(human_accuracy, machine_accuracy) 
                n_data = int(conf.data_size*split)
                entries.append((n_data,  gain*100))


    df = pd.DataFrame(entries, columns=[r'$m$', r'$\textsc{Success probability gain  }\%$' ])
    
    mean_df_m = df.groupby([ r'$m$']).mean()
    std_df_m = df.groupby([ r'$m$']).std()/np.sqrt(30)

    return mean_df_m, std_df_m


def real_models_acc(split):
    """Test set accuracy of pre-trained models"""
    base  = f"results_real/calibrationSet{split}"
    d = []
    for machine_model in conf.model_names:
        best = []
        for run in range(10):
            dir = f"{base}\{machine_model}_run{run}"
            with open(f'{dir}/logs.txt', 'r') as f:
                acc = float(f.readlines()[3])
                d.append((machine_model, acc))

    df = pd.DataFrame(d, columns=["model", "acc"])
    return df.groupby("model").mean()

def real_human_acc():
    """Human test set accuracy in real data experiments"""
    err = []
    for r in range(10):
        X_test, X_cal, y_test, y_cal = utils.make_dataset_real(r)                
        conf.accuracy = None
        human = ExpertReal(conf)
        p = torch.tensor(human.confusion_matrix[y_test], device=conf.device)
        y_hat = p.multinomial(1, replacement=True, generator=conf.torch_rng).detach().cpu().numpy().flatten() 
        err.append(np.mean((y_test != y_hat)))
    return 1 - np.mean(err)