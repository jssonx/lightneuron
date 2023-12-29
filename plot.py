import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Set the aesthetic style of the plots
sns.set(style="ticks", palette="pastel")

# Path to the 'bench' directory
bench_path = './bench'

# Initialize a DataFrame to store all data
all_data = pd.DataFrame()

# List all files in the 'bench' directory and read their contents into a DataFrame
for filename in os.listdir(bench_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(bench_path, filename)
        # Read data from each file
        data = pd.read_csv(file_path, header=None, names=['N', filename], index_col=0)
        # Interpolate missing values for a smoother line
        data = data.reindex(range(data.index.min(), data.index.max())).interpolate()
        all_data = pd.concat([all_data, data], axis=1)

# Sort columns by the performance at N=1000 (or the closest available value)
# We take the value for N=1000 or interpolate if it doesn't exist
performance_at_1000 = all_data.loc[all_data.index.get_loc(1000, method='nearest')].sort_values(ascending=False)
sorted_columns = performance_at_1000.index

# Plot the data
fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the size of the figure

# Generate a color palette that's visually distinct
colors = plt.cm.tab20(np.linspace(0, 1, len(sorted_columns)))

# Plot each column with the sorted colors
for col, color in zip(sorted_columns, colors):
    ax.plot(all_data.index, all_data[col], label=col, color=color)

# Customize the plot to make it publication-quality
ax.set_title('GFLOPS Performance by Benchmark', fontsize=16, weight='bold')
ax.set_xlabel('Matrix Size N', fontsize=14, weight='bold')
ax.set_ylabel('GFLOPS', fontsize=14, weight='bold')

# Adjust the position of the legend and the plot
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])  # Resize plot to make space for the legend
legend = ax.legend(title='Benchmarks', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.setp(legend.get_title(), fontsize='12')  # Set the fontsize of the legend's title

ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.tick_params(axis='both', which='major', labelsize=12)

# Set the tick labels to be outside
sns.despine(trim=True, offset=10)

# Save the figure
fig_path = 'gflops_performance.png'
plt.savefig(fig_path, format='png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

# Return the path to the saved figure
fig_path