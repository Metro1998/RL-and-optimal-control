import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


def visualize(trajectories_overall, cost, file_path_for_pic):
    """
    Visualize the trajectories.

    :param trajectories_overall:
    :param cost:
    :param file_path_for_pic:
    :return:
    """
    agent_to_color_dictionary = {
        'continuous': '#0000FF',
        'discrete': '#FF69B4',
        'hybrid': '#800080',
        'pattern_1': '#0000FF',
        'pattern_2': '#FF69B4',
        'pattern_3': '#800080',
    }

    fig, ax = plt.subplots()
    ax.set_facecolor('xkcd:white')
    ax.set_ylabel('X_1', loc='top', labelpad=-2, fontdict={'font': 'Times New Roman', 'fontsize': 8})
    ax.set_xlabel('X_0', loc='right', labelpad=-2, fontdict={'font': 'Times New Roman', 'fontsize': 8})
    for spine in ['right', 'top']:
        ax.spines[spine].set_visible(False)

    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))

    # preprocess of the data
    trajectories_executed = trajectories_overall[:, 0, :].squeeze().transpose()
    trajectories_overall = trajectories_overall

    """# mark the statistic
    indicator = 0
    for x_0, x_1 in zip(trajectories_executed[0], trajectories_executed[1]):
        if x_0 <= 0 and x_1 >= 0:
            a, b = -0.1, 0.1
        elif x_0 <= 0 and x_1 <= 0:
            a, b = -0.1, -0.1
        elif x_0 >= 0 and x_1 >= 0:
            a, b = 0.1, 0.1
        else:
            a, b = 0.1, -0.1
        if abs(x_0) < 0.01 and abs(x_1) < 0.01 and indicator == 1:
            pass
        else:
            if abs(x_0) < 0.01 and abs(x_1) < 0.01:
                indicator = 1
            plt.text(x_0 + a, x_1 + b, '({}, {})'.format(round(x_0, 2), round(x_1, 2)), fontdict={'font': 'Times New Roman', 'fontsize': 8})"""

    # plot all predictive trajectories
    for i in range(trajectories_overall.shape[0]):
        tra = trajectories_overall[i].transpose()
        if i == 0:
            ax.plot(tra[0], tra[1], label='predictive trajectories', color='#FF69B4', marker='o', markersize=2, alpha=0.6)
        else:
            ax.plot(tra[0], tra[1], color='#FF69B4', marker='o', markersize=2, alpha=0.6)

    # plot executed trajectories
    ax.plot(trajectories_executed[0], trajectories_executed[1],
            label='executed trajectories', color='#800080', marker='s', markersize=2)
    # mark its cumulative cost
    for x_0, x_1, c in zip(trajectories_executed[0], trajectories_executed[1], cost):
        plt.text(x_0, x_1, '{}'.format(round(c[0]), 2), fontdict={'font': 'Times New Roman', 'fontsize': 8})

    ax.legend(loc='upper right', frameon=False)

    ax.set_xlim([-6, 6])
    ax.set_ylim([-6, 6])

    xlabels = ['-5', '-4', '-3', '-2', '-1', ' ', '1', '2', '3', '4', '5']
    ylabels = ['-5', '-4', '-3', '-2', '-1', ' ', '1', '2', '3', '4', '5']
    ax.set_xticks(np.arange(len(xlabels)) - 5, xlabels)
    ax.set_yticks(np.arange(len(ylabels)) - 5, ylabels)

    x1_label = ax.get_xticklabels()
    [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
    y1_label = ax.get_yticklabels()
    [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

    ax.tick_params(axis='both',
                   labelsize='medium',  # y轴字体大小设置
                   width=1,
                   length=1.5,
                   direction='out'
                   )
    legend_font = {
        'family': 'Times New Roman',  # 字体
    }

    ax.legend(
        prop=legend_font,
        frameon=False
    )

    plt.tight_layout()
    plt.savefig(file_path_for_pic, dpi=600)