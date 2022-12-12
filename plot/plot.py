import numpy as np
import pandas as pd
import seaborn as sns
from config import conf
import matplotlib.pyplot as plt
import matplotlib as mpl
from plot.utils import *
import os

legend_font_size = 32
marker_size_main = 250
marker_size_hat = 200
markerscale = 4

mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amsfonts,geometry}'
mpl.rcParams['axes.formatter.use_mathtext'] = True
plt.rcParams.update({
    'font.family':'serif',
    "font.serif": ["Computer Modern Roman"],
    "text.usetex": True,
})

# Create path to store plots
plot_path_root = f"{conf.ROOT_DIR}/pdfs"
if not os.path.exists(plot_path_root):
    os.mkdir(plot_path_root)

def alpha_vs_success_synthetic_four_confs(labels=10, acc1=0.5, acc2=0.9, split=0.15, run=0, results_root='results_synthetic'):
    """Plot alpha vs empirical expert success probability for 2 experts and 2 classifiers on synthetic data""" 
    plt.rcParams.update({
    "font.size": 38,
    "figure.figsize":(15,9)
    })
    acc_a_to_df = []
    marker_x = []
    marker_y = []
    human_str = r'$\mathbb{P}[\hat Y = Y \,;\, \mathcal{Y}] = '
    machine_str = r'\mathbb{P}[Y^\prime = Y] = '
    for human_accuracy,machine_accuracy in [(acc1,acc2),(acc2,acc2),(acc2,acc1), (acc1,acc1)]:
        base = f"{results_root}/{labels}labels_calibrationSet{split}"
        dir = f"{base}/human{human_accuracy}_machine{machine_accuracy}_run{run}"
        keys = ['alphas1', 'alpha1_idx', 'alpha1_test_error']
        data = read_specific_data(dir, keys)
        # markers for ^alpha
        marker_x.append(data['alphas1'][data['alpha1_idx']])
        marker_y.append(1 - data['alpha1_test_error'][data['alpha1_idx']])
        
        for a, err in list(zip(data['alphas1'], data['alpha1_test_error'])):
            acc_a_to_df.append((f"{human_str}{human_accuracy}, {machine_str}{machine_accuracy}$", a, 1 - err))

    df = pd.DataFrame(acc_a_to_df, columns=["human_machine", r'$\alpha$', 'Empirical Success Probability'])
   
    markers = {f"{human_str}{acc1}, {machine_str}{acc2}$" : 'X',
                f"{human_str}{acc2}, {machine_str}{acc2}$": 'X',
                f"{human_str}{acc2}, {machine_str}{acc1}$": 'o',
                f"{human_str}{acc1}, {machine_str}{acc1}$": 'o'}
    ax = sns.scatterplot(data=df, x=r'$\alpha$', y='Empirical Success Probability', 
                         hue="human_machine", style="human_machine", 
                         markers=markers, edgecolor=None, palette='colorblind',
                         s=marker_size_main, rasterized=True)
    ax.yaxis.labelpad = 20
    ax.legend(fontsize=legend_font_size, loc='lower left', markerscale=markerscale)

    c = ['darkblue','saddlebrown','darkgreen','darkred']
    ax.scatter(x=marker_x, y=marker_y, color=c, marker='s', s=marker_size_hat)
    
    plt.savefig(f"{plot_path_root}/{acc1}_{acc2}_success_vs_alpha_sythetic.pdf")

def alpha_vs_success_real(split=0.15, run=0, results_root='results_real'):
    """Plot of empirical expert success probability vs alpha in real data experiments"""
    plt.rcParams.update({
    "font.size": 38,
    "figure.figsize":(15,9)
    })
    acc_a_to_df = []
    marker_points = []
    values = [r'$\textsc{DenseNet-BC}$', r'$\textsc{PreResNet-110}$', r'$\textsc{ResNet-110}$']
    model_name_mapping = dict(zip(conf.model_names, values))
    base = f"{results_root}/calibrationSet{split}"
    marker_x = []
    marker_y = []
    
    for machine_model in reversed(conf.model_names):
        dir = f"{base}/{machine_model}_run{run}"     
        keys = ['alphas1', 'alpha1_idx', 'alpha1_test_error']
        data = read_specific_data(dir, keys)
        alphas_sorted = np.unique(sorted(data['alphas1']))
        alpha_hat = data['alphas1'][data['alpha1_idx']]

        marker_points.append(np.argwhere(alphas_sorted==alpha_hat)[0][0])
        marker_x.append(alpha_hat)
        marker_y.append(1 - data['alpha1_test_error'][data['alpha1_idx']])
        
        for a, err in list(zip(data['alphas1'], data['alpha1_test_error'])):
            acc_a_to_df.append((f"{model_name_mapping[machine_model]}", a, 1 - err))

    df = pd.DataFrame(acc_a_to_df, columns=["model", r'$\alpha$', 'Empirical Success Probability'])
    ax = sns.scatterplot(data=df, x=r'$\alpha$', y='Empirical Success Probability',
                         hue="model", style='model', palette='colorblind', 
                         s=marker_size_main, edgecolor=None,rasterized=True)
    ax.yaxis.labelpad = 20
    ax.legend(fontsize=legend_font_size, loc='lower left',  markerscale=markerscale)
    ax.scatter(x=marker_x, y=marker_y, color=["darkblue","saddlebrown","darkgreen"], marker='s', s=marker_size_hat)    
    ax.set(xscale='log')

    results_type = '_'.join(results_root.split('/')[-1].split('_')[1:])
    plt.savefig(f"{plot_path_root}/success_vs_alpha_{results_type}.pdf")

def alpha_vs_avg_size_synthetic_four_confs(labels=10, acc1=0.5, acc2=0.9, split=0.15, run=0, results_root='results_synthetic'):
    """Plot alpha vs empirical average set size for 2 experts and 2 classifiers with synthetic data"""
    plt.rcParams.update({
    "font.size": 38,
    "figure.figsize":(15,9)
    })

    size_a_to_df = []
    marker_x = []
    marker_y = []
    machine_str = r'$\mathbb{P}[Y^\prime = Y] = '
    for machine_accuracy in [acc1,acc2]:
        for human_accuracy in [acc1,acc2]:
            base = f"{results_root}/{labels}labels_calibrationSet{split}"
            dir = f"{base}/human{human_accuracy}_machine{machine_accuracy}_run{run}"            
            keys = ['alphas1', 'alpha1_idx', 'alpha1_avg_set_size_test']
            data = read_specific_data(dir, keys)

            marker_x.append(data['alphas1'][data['alpha1_idx']])
            marker_y.append(data['alpha1_avg_set_size_test'][data['alpha1_idx']])
        
        for a, s in list(zip(data['alphas1'], data['alpha1_avg_set_size_test'])):
            size_a_to_df.append((f"{machine_str}{machine_accuracy}$", a, s))

    exp_size_str = 'Empirical Average Set Size'
    df = pd.DataFrame(size_a_to_df, columns=["human_machine",  r'$\alpha$', exp_size_str])
    palette = {f"{machine_str}{acc1}$": 'lightgray',f"{machine_str}{acc2}$": 'darkgray'}
    ax = sns.scatterplot(data=df, x=r'$\alpha$', y=exp_size_str, hue="human_machine",
                         hue_order=[f"{machine_str}{acc2}$", f"{machine_str}{acc1}$"],
                         edgecolor=None, style="human_machine", palette=palette, 
                         s=marker_size_main, rasterized=True)
    ax.yaxis.labelpad = 20
    ax.legend(fontsize=legend_font_size, loc='upper right', markerscale=markerscale)

    c = ['darkred','darkgreen','darkblue','saddlebrown']
    ax.scatter(x=marker_x, y=marker_y, c=c, marker='s', s=marker_size_hat)

    plt.savefig(f"{plot_path_root}/{acc1}_{acc2}_avg_set_size_vs_alpha_sythetic.pdf")

def alpha_vs_avg_size_real(split=0.15, run=0, results_root='results_real'):
    """Plot of empirical average set size vs alpha in real data experiments"""
    plt.rcParams.update({
    "font.size": 38,
    "figure.figsize":(15,9)
    })
    acc_a_to_df = []
    values = [r'$\textsc{DenseNet-BC}$', r'$\textsc{PreResNet-110}$', r'$\textsc{ResNet-110}$']
    model_name_mapping = dict(zip( conf.model_names, values))
    base = f"{results_root}/calibrationSet{split}"
    marker_x = []
    marker_y = []
   
    for machine_model in reversed(conf.model_names):
        dir = f"{base}/{machine_model}_run{run}"
        keys = ['alphas1', 'alpha1_idx', 'alpha1_avg_set_size_test']
        data = read_specific_data(dir, keys)

        marker_x.append(data['alphas1'][data['alpha1_idx']])
        marker_y.append(data['alpha1_avg_set_size_test'][data['alpha1_idx']])

        for a, s in list(zip(data['alphas1'], data['alpha1_avg_set_size_test'])):
            acc_a_to_df.append((f"{model_name_mapping[machine_model]}", run, a, s))
    
    exp_size_str = 'Empirical Average Set Size'
    df = pd.DataFrame(acc_a_to_df, columns=["model",'run', r'$\alpha$', exp_size_str])
    ax = sns.scatterplot(data=df, x=r'$\alpha$', y=exp_size_str, hue="model",
                         style='model', palette='colorblind', s=marker_size_main, 
                         edgecolor=None, rasterized=True)
    ax.yaxis.labelpad = 20
    ax.legend(fontsize=legend_font_size, loc='upper right', markerscale=markerscale)
    ax.scatter(x=marker_x, y=marker_y,c=["darkblue","saddlebrown","darkgreen"], marker='s', s=marker_size_hat)
    ax.set(xscale='log')
   
    results_type = '_'.join(results_root.split('/')[-1].split('_')[1:])
    plt.savefig(f"{plot_path_root}/avg_set_size_vs_alpha_{results_type}.pdf")

def set_size_distr_synthetic(human_acc=0.5, machine_acc=0.5, split=0.15, labels=10, runs=10, results_root="results_synthetic"):
    """Plot of subset size distribution for a given synthetic task and an expert"""
    plt.rcParams.update({
    "font.size": 80,
    "figure.figsize":(16,10)
    })
    accs_distr_tuple_list = []
    plt.figure()
    base=f"{results_root}/{labels}labels_calibrationSet{split}"
    for run in range(runs):
        dir = f"{base}/human{human_acc}_machine{machine_acc}_run{run}"
        keys = ['alpha1_set_size_distr_dict']
        set_size_dict = read_specific_data(dir, keys)[keys[0]]
        for size,count in set_size_dict.items():
            accs_distr_tuple_list.append((size, count, run))

    df = pd.DataFrame(accs_distr_tuple_list, columns=["Set-size", "count", "run"])
    ax = sns.barplot(data=df, y='count', x='Set-size', color=sns.color_palette('colorblind')[0])
    ax.set_xlabel(r'$|\mathcal{C}_{\hat{\alpha}}(X)|$')
    ax.set_ylabel(r'Frequency')

    plt.savefig(f"{plot_path_root}/human{human_acc}_machine{machine_acc}_set_size_distr_synthetic.pdf",bbox_inches='tight')

def coverage(synthetic=True, split=0.15, labels=10, results_root='results_coverage'):
    """Plots on empirical vs target coverage during test"""
    plt.rcParams.update({
    "font.size": 50,
    "legend.fontsize": 50,
    "figure.figsize":(16,10)
    })
    entries = []
    
    root_dir = f"{results_root}_{'synthetic' if synthetic else 'real'}"  
    base_dir = f"{root_dir}/{f'{labels}labels_'if synthetic else ''}calibrationSet{split}"

    for root, dirs, files in os.walk(base_dir):
        for r,dir in enumerate(dirs):
            keys = ['alpha1_emp_coverage', 'alpha1_value']
            data = read_specific_data(f"{root}/{dir}", keys)
            entries.append((r, float(data['alpha1_emp_coverage']),"Empirical" ))
            target_cov = 1 - data['alpha1_value'] 
            entries.append((r, target_cov, "Target"))

    df_values = pd.DataFrame(entries, columns=["Run","Coverage Value","Coverage Type"])
    df = df_values
    palette = sns.color_palette('colorblind')
    
    style = {'Empirical': {'color':palette[0], 'linestyle':'-', 'marker':'.', 'linewidth':4, 'markersize':25},
             'Target':{'color':palette[1], 'linestyle':'--', 'marker':'x', 'linewidth':4, 'markersize':25}}

    main_df = df[(df['Coverage Type'] == 'Empirical') | (df['Coverage Type'] == 'Target')]
    ax = sns.scatterplot(data=main_df, x='Run', y='Coverage Value', s=200,  style='Coverage Type',hue='Coverage Type', palette=palette[:2])
    ax = sns.lineplot(data=df,y="Coverage Value", x='Run', hue="Coverage Type",  style='Coverage Type', linewidth=4, palette=palette[:2], markers=False, ax=ax)
    ax.set_xlabel('Index')
    ax.set_ylabel('Coverage')
    ax.yaxis.labelpad = 20
    ax.legend().set_title('')
    def create_dummy_line(**kwds):
        return mpl.lines.Line2D([], [], **kwds)

    ax.legend([create_dummy_line(**v) for v in style.values()] , [r'$1 - \alpha_{\text{emp}}$', r'$1 - \hat{\alpha}$' ])
    if synthetic:
        ax.set_ylim([0.7, 1.])
    else:
        ax.set_ylim([0.98, 1.])
   
    plt.savefig(f"{plot_path_root}/coverage_{'synthetic' if synthetic else 'real'}.pdf", transparent=True, bbox_inches='tight')

def robustness_synthetic(split=.15, labels=10, results_root='results_synthetic', runs=10, human_accuracy=0.5, machine_accuracies=[.5,.7,.9]):
    """Robustness plots for a given expert on three synthetic tasks with 3 classifiers"""
    plt.rcParams.update({
    "font.size": 80,
    "legend.fontsize": 60,
    "figure.figsize":(16,10)
    })
    entries = []
    plt.figure()
    
    base  = f"{results_root}/{labels}labels_calibrationSet{split}"
    keys = ['alpha1_idx', 'alpha1_test_error', 'alpha1_robustness_error_dict']
    get_success_prob = lambda data,_ : (1 - data['alpha1_test_error'][data['alpha1_idx']], data['alpha1_robustness_error_dict'],)
    entries_mixed = get_synthetic_results(base, keys, runs, get_success_prob, human_accs=[human_accuracy]) 
    entries = []
    for h_acc, m_acc, success_prob_ref, robustness_error_dict, run in entries_mixed:
        machine_str = r'$\mathbb{P}[Y^\prime= Y] = $'+f" {m_acc}"
        entries.append((0, success_prob_ref, machine_str, run))                
        for p, error in robustness_error_dict.items():
            entries.append((p, 1 - error,  machine_str, run)) 

    df = pd.DataFrame(entries, columns=[r'$p$',  "Success Probability",  "machine", "run"])
    palette = sns.color_palette('colorblind')
    ax = sns.lineplot(data=df, x=r'$p$',y='Success Probability', hue='machine', style='machine', palette=palette[:len(machine_accuracies)])
    ax.legend().set_title('')
    ax.axhline(human_accuracy, ls='--', color='black', label='')

    ax.set_xticks(np.arange(0, 1.01, .2))
    ax.set_ylabel('Empirical\n Success Probability')
    ax.yaxis.labelpad = 10
    ax.get_legend().remove()
    plt.savefig(f"{plot_path_root}/robustness_synthetic_human{human_accuracy}.pdf", transparent=True, bbox_inches='tight')

def robustness_real(split=.15, results_root='results_real', runs=10):
    """Robustness plots for real data experiments"""
    plt.rcParams.update({
    "font.size": 50,
    "legend.fontsize": 40,
    "figure.figsize":(16,10)
    })
    entries = []
    values = [r'$\textsc{DenseNet-BC}$', r'$\textsc{PreResNet-110}$',r'$\textsc{ResNet-110}$']
    model_name_mapping = dict(zip( conf.model_names, values))
    base  = f"{results_root}/calibrationSet{split}"
    keys = [f"alpha1_robustness_error_dict"]
    get_pvalue_vs_acc = lambda data, _ : ([(p, 1 - error) for p, error in data[keys[0]].items()],)
    entries_folded = get_real_results(base, keys, runs, get_pvalue_vs_acc)
    for machine_model, pval_acc_list, run in entries_folded:
        for p, acc in pval_acc_list:
            entries.append((model_name_mapping[machine_model], p, acc, run))

    df = pd.DataFrame(entries, columns=['model', r'$p$',  "Success Probability", "run"])
    palette = sns.color_palette('colorblind')
    order = [model_name_mapping[machine_model] for machine_model in conf.model_names]
    ax = sns.lineplot(data=df, x=r'$p$',y='Success Probability', hue='model',style='model', palette=palette[:3], hue_order=order, style_order=order)
    ax.legend().set_title('')
    ax.set_ylabel('Empirical\n Success Probability')
    ax.yaxis.labelpad = 20
    ax.legend(loc='lower left')

    plt.savefig(f"{plot_path_root}/robustness_real.pdf", transparent=True, bbox_inches='tight')

def topk_synthetic(kmin=2, kmax=9, split=0.15, labels=10, runs=10, human_accuracy=0.5, machine_accuracy=0.9, results_root="results_synthetic"):
    """Plot for top-k predictors for one expert in one synthetic task with one classifier"""
    plt.rcParams.update({
    "font.size": 80,
    "figure.figsize":(16,10)
    })
    plt.figure()

    topk_success_probs = []
    for k in range(kmin, kmax+1):
        base  = f"{results_root}_top{k}/{labels}labels_calibrationSet{split}"
        keys = [f"top{k}_test_error"]
        get_success_prob = lambda data,_: (1 - data[keys[0]],)
        entries = get_synthetic_results(base, keys, runs, get_success_prob, human_accs=[human_accuracy], machine_accs=[machine_accuracy])
        topk_success_probs.extend([(topk_success_prob, k, run) for _, _, topk_success_prob, run in entries])
    
    base = f"{results_root}/{labels}labels_calibrationSet{split}"
    keys = ['alpha1_idx', 'alpha1_test_error']
    get_success_prob = lambda data, _ :(1 - data['alpha1_test_error'][data['alpha1_idx']], )
    entries = get_synthetic_results(base, keys, runs, get_success_prob, human_accs=[human_accuracy], machine_accs=[machine_accuracy])
    cp_success = [(succes_prob, f"Standard CF", run) for _, _, succes_prob, run in entries]

    df_ref = pd.DataFrame(cp_success, columns=["Success Probability", "method", "run"])    
    df = pd.DataFrame(topk_success_probs, columns=["Success Probability", "method", "run"])
    
    palette = sns.color_palette('colorblind')
    grey = palette[7]
    red = palette[3]
    optimal_k_array = df.pivot_table(index='method', values='Success Probability').idxmax().to_numpy()
    optimal_k = optimal_k_array[0]
    colors = [grey if k != optimal_k else red for k in range(kmin, kmax+1)]
        
    ax = sns.barplot(data=df, x='method', y='Success Probability', palette=colors)
    ours_mean_value = df_ref['Success Probability'].mean()
    
    ymin = df['Success Probability'].min() 
    ymax = 1.
    ax.set_ylim((ymin,ymax))
    ax.set_yticks(np.arange(0.6,0.9,0.1))
    ax.set_ylabel('Empirical\n Success Probability')
    ax.set_xlabel(r'$k$')
    ax.axhline(ours_mean_value, ls='--', color='black', label='Ours')
    ax.yaxis.labelpad = 20

    plt.savefig(f"{plot_path_root}/topk_synthetic_human{human_accuracy}_machine{machine_accuracy}_red.pdf",transparent=True, bbox_inches='tight')

def topk_real(kmin=2, kmax=9, split=0.15, runs=10, results_root="results_real"):
    """Plot for top-k predictors for real data experiments"""
    plt.rcParams.update({
    "font.size": 80,
    "figure.figsize":(16,10)
    })
    topk_errors = []
    for k in range(kmin, kmax+1):
        base = f"{results_root}_top{k}/calibrationSet{split}"
        keys = [f"top{k}_test_error"]
        get_topk_error = lambda data,_: (1 - list(data.values())[0], )
        errors = get_real_results(base, keys, runs, get_topk_error)
        topk_errors.extend(list(t+(k,) for t in errors))
    
    base = f"{results_root}/calibrationSet{split}"
    cf_errors = []
    keys = ['alpha1_idx', 'alpha1_test_error']
    get_cf_error = lambda data,_: (1 - data[keys[1]][data[keys[0]]] ,)
    errors = get_real_results(base, keys, runs, get_cf_error)
    cf_errors.extend(list(t+(f"Standard CF",) for t in errors))

    df_ref = pd.DataFrame(cf_errors, columns=["Machine", "Success Probability", "run", "method"])  
    df = pd.DataFrame(topk_errors, columns=["Machine", "Success Probability", "run", "method"])

    for model_name in conf.model_names:
        plt.figure()
        df_model = df[df['Machine'] == model_name]
        df_ref_model = df_ref[df_ref['Machine'] == model_name]
        palette = sns.color_palette('colorblind')
        grey = palette[7]
        red = palette[3]
        optimal_k_str_array = df_model.pivot_table(index='method', values='Success Probability').idxmax().to_numpy()
        optimal_k = int(optimal_k_str_array[0])
        colors = [grey if k != optimal_k else red for k in range(kmin, kmax+1)]
        ax = sns.barplot(data=df_model, x='method', y='Success Probability', palette=colors)
        ours_mean_value = df_ref_model['Success Probability'].mean()
        ymin = df_model['Success Probability'].min() 
        ymax = 1.
        ax.set_ylim((ymin,ymax))
        ax.set_yticks(np.arange(0.9,1, 0.02))
        ax.set_ylabel('Empirical\n Success Probability')
        ax.set_xlabel(r'$k$')
        ax.yaxis.labelpad = 20
        ax.axhline(ours_mean_value, ls='--', color='black', label='Ours')
            
        plt.savefig(f"{plot_path_root}/topk_real_{model_name}_red.pdf", transparent=True, bbox_inches='tight')
