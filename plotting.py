import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_temp_histograms(datasets, titles):
    N = len(datasets)
    fig, axes = plt.subplots(1, N, figsize=(1+3*N, 3), sharex=True, sharey=True)
    if N == 1:
        axes = [axes]
    for data, title, ax in zip(datasets, titles, axes):
        sns.histplot(data.p_boltz_2ary, bins=50, ax=ax)
        ax.set_xlabel('p_boltz')
        ax.set_ylabel('Count')
        ax.set_title(title)
    plt.tight_layout()
    plt.show()

def plot_temp_histogram(data):
    plot_temp_histograms([data], [''])

def plot_model_probs(data, model, model_name, ax=None, type='hist', eval_temp=1e-2):
    pref_classes = {
        '>': data.y == 1,
        '~': data.y == 0.5,
        '<': data.y == 0,
    }
    fixed_T = eval_temp * np.ones(len(data.x1))
    p_hat = model[0].predict((data.x1, data.x2, fixed_T), batch_size=32)

    bins = np.linspace(0, 1, 30)

    p_hat_by_type = {
        key: p_hat[idx].squeeze() for key, idx in pref_classes.items()
    }

    should_show_plot = False
    if ax is None:
        should_show_plot = True
        fig, ax = plt.subplots()
    ax.set_title('Model: ' + model_name)
    ax.set_xlabel('Probability')
    if type == 'hist':
        sns.histplot(p_hat_by_type, alpha=0.7, palette=['C0', 'C8', 'C3'], bins=bins, kde=False, stat="count", ax=ax, legend=True)
        ax.set_ylabel('Count')
        ax.get_legend().set_loc('upper center' if 'rational' not in model_name else 'best')
    elif type == 'kde':
        sns.kdeplot(p_hat_by_type, alpha=0.7, palette=['C0', 'C8', 'C3'], ax=ax, common_norm=False, clip=[0,1])
        ax.set_ylabel('Density')
    else:
        raise ValueError(f"Unknown 'type' parameter: {type}")
    if should_show_plot:
        ax.show()

def plot_utility_calibration(data, model, model_name, eval_temp=1e-2, ax=None):
    fixed_T = eval_temp * np.ones(len(data.x1))
    u1_hat, _ = model[1].predict((data.x1, data.x2, fixed_T), batch_size=32)

    should_show_plot = False
    if ax is None:
        should_show_plot = True
        fig, ax = plt.subplots()
    ax.set_title('Model: ' + model_name)
    ax.set_xlabel('Actual utility')
    sns.boxenplot(x=9*data.u1, y=9*u1_hat.squeeze(), ax=ax, legend=True)
    ax.set_ylim([0,9])
    ax.set_ylabel('Predicted utility')
    if should_show_plot:
        ax.show()

def plot_probability_calibration(data, model, model_name, ax=None):
    p_hat = model[0].predict((data.x1, data.x2, data.T), batch_size=32)

    should_show_plot = False
    if ax is None:
        should_show_plot = True
        fig, ax = plt.subplots()
    ax.set_title('Model: ' + model_name)
    ax.set_xlabel(r'Actual $P_{Boltz}$')
    sns.kdeplot(x=data.p_boltz_2ary, y=p_hat.squeeze(), clip=[0,1], ax=ax, legend=True)
    ax.set_ylim([0,1])
    ax.set_ylabel(r'Predicted $\hat P_{Boltz}$')
    if should_show_plot:
        ax.show()
