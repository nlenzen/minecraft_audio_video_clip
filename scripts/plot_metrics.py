import matplotlib.pyplot as plt
import numpy as np

# plt.title(f'Max Item Counts For "{task.prompt}"')
    # plt.xlabel('Module Configurations')
plt.ylabel(f'{task.get_item_name_str()} {"Placed" if task.is_place_task else "Collected"}')

plt.bar(module_config_xlabels, module_config_metrics,
        color=['steelblue', 'darkgoldenrod', 'darkcyan', 'chocolate'], zorder=2)
plt.grid(axis='y', linestyle='-')
x_pos = np.arange(len(module_config_xlabels))
plt.errorbar(x_pos, module_config_metrics, yerr=module_config_error_bars,
             fmt='none', ecolor='black', capsize=3)

plt.xticks(rotation=0, ha='center', wrap=True)
# plt.xticks(rotation=-45, ha='left', wrap=True)  # TODO: Change later
plt.tight_layout()

# Increas eall font size
plt.rcParams.update({'font.size': 18})
plt.rcParams['mathtext.fontset'] = 'stix'

# Adapt the height so that the bottom isnt cut off
# plt.subplots_adjust(bottom=0.25)

plt.savefig(plot_save_filepath)
plt.clf()
plt.close()