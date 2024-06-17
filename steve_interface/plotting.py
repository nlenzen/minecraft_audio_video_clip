import os
import numpy as np
import matplotlib.pyplot as plt
from minecraft_audio_video_clip.steve_interface.load_util import load_json



def get_values(path):
    results = []
    similarities = []
    for file in os.listdir(path):
        if file.endswith('.json'):
            file_path = os.path.join(path, file)
            data = load_json(file_path)
            results.append(data['count'])
            sim = data.get('max_similarity', None)
            if sim is not None:
                similarities.append(sim)

    return results, similarities


def plot_task_results(path, prompt_types, colors, savepath=None, graph_labels=None, list_values=False):
    task_name = os.path.basename(os.path.normpath(path))
    path = os.path.join(path, 'metrics')

    values = []
    similarities = []
    for prompt_type in prompt_types:
        metric_path = os.path.join(path, prompt_type)
        vals, sim = get_values(metric_path)
        values.append(vals)
        if list_values:
            print(f'{prompt_type}: {vals}')
        if len(sim) > 0:
            similarities.append(sim)
    values = np.array(values)
    if len(similarities) > 0:
        similarities = np.array(similarities)
        mean_sim = np.mean(similarities, axis=1)
        print(f'Mean similarities: {mean_sim}')

    mean = np.mean(values, axis=1)
    print(f'Mean values: {mean}')

    # Compute confidence Interval
    std = np.std(values, axis=1)
    error = 1.96 * std / np.sqrt(values.shape[1])    # 1.96 is the constant for a 95% confidence interval

    # plotting
    if savepath is not None:
        plt.rcParams.update({'font.size': 16})
        # plt.rcParams.update({'font.weight': 'bold'})
        ylabel = task_name.replace('_', ' ').title()
        plt.ylabel(ylabel, fontsize=18)
        labels = graph_labels if graph_labels is not None else prompt_types
        plt.bar(labels, mean, color=colors, zorder=2)
        plt.grid(axis='y', linestyle='-')
        pos = np.arange(len(prompt_types))
        plt.errorbar(pos, mean, yerr=error, fmt='none', ecolor='black', capsize=3)

        plt.xticks(rotation=0, ha='center', wrap=True)
        plt.tight_layout()

        plt.rcParams.update({'font.size': 18})
        plt.rcParams['mathtext.fontset'] = 'stix'

        os.makedirs(savepath, exist_ok=True)
        savepath = os.path.join(savepath, task_name + '.png')
        plt.savefig(savepath)

        plt.clf()
        plt.close()


