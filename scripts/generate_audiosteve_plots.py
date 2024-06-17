from minecraft_audio_video_clip.steve_interface.plotting import plot_task_results
import os

# path = 'videos/evaluation/destroy_leaves'
path = 'videos/cvae_evaluation'
savepath = 'results/audio_steve/audio_comparison'
# savepath = 'results/audio_steve/'
# savepath = None

# metrics = ['visual', 'text', 'audio']
metrics = ['audio_base_old', 'audio_base', 'audio_old', 'audio']
# metrics = ['audio_base', 'audio_one', 'audio_two', 'audio_old', 'audio']
colors = ['steelblue', 'darkgoldenrod', 'darkcyan', 'chocolate']

graph_labels = ['Audio\nBase\nOld', 'Audio\nBase\nNew', 'Audio\nPrompt\nOld', 'Audio\nPrompt\nNew']

for task in os.listdir(path):
    print(f'Task {task}')
    task_path = os.path.join(path, task)
    plot_task_results(task_path, metrics, colors, savepath,  graph_labels=graph_labels, list_values=False)

