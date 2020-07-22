import matplotlib.pyplot as plt


def plot_cal_result(r_dict, labels: bool = True):
    r_vals = list(r_dict.values())
    gens = list(r_dict.keys())
    r_vals, gens = list(zip(*sorted(zip(r_vals, gens))))
    fig, ax = plt.subplots()
    x = list(range(len(r_dict)))
    ax.barh(x, r_vals)
    if labels:
        ax.set_yticks(range(len(gens)))
        ax.set_yticklabels(gens)
    else:
        ax.set_yticks([])
    ax.set_xlabel('Generator coefficient r_i')
    ax.set_ylabel('Generator G_i')
    return fig, ax
