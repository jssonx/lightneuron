import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np

####################################################################################################
# PLOT 1: PERFORMANCE COMPARISON
####################################################################################################

# Set the aesthetic style of the plots
sns.set(style="ticks", palette="pastel")

# Path to the 'bench' directory
bench_path = './bench'

# Initialize a list to store indices from each file
indices = []

# Collect indices from each file
for filename in os.listdir(bench_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(bench_path, filename)
        data = pd.read_csv(file_path, header=None, index_col=0)
        indices.append(data.index.tolist())

# Find the common indices in all files
common_indices = set(indices[0]).intersection(*indices[1:])

# Convert the set of common indices to a sorted list
common_indices_list = sorted(list(common_indices))

# Initialize a DataFrame to store all data
all_data = pd.DataFrame(index=common_indices_list)

# Read and combine data from each file
for filename in os.listdir(bench_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(bench_path, filename)
        column_name = filename.replace('.txt', '')
        data = pd.read_csv(file_path, header=None, names=[column_name], index_col=0)
        all_data = all_data.join(data, how='inner')

# Sort the DataFrame index
all_data.sort_index(inplace=True)

# Plot the data
fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the size of the figure

# Sort columns by their performance at the largest common N value
sorted_columns = all_data.loc[max(all_data.index)].sort_values(ascending=False).index

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
fig_path = './img/gflops_performance_test.png'
plt.savefig(fig_path, format='png', dpi=300, bbox_inches='tight')

####################################################################################################
# PLOT 2: PERFORMANCE RELATIVE TO OpenBLAS
####################################################################################################

# Get the data at matrix size 1200x1200
data_at_1200 = all_data.loc[1200]

# Benchmark: OpenBLAS
openblas_gflops = data_at_1200['OpenBLAS']

# Construct a Markdown table
markdown_table = "GFLOPs at matrix size 1200x1200:\n"
markdown_table += "<!-- benchmark_results -->\n"
markdown_table += "| Kernel | GFLOPs/s | Performance relative to OpenBLAS |\n"
markdown_table += "|:-------|---------:|:-------------------------------|\n"

for kernel in sorted_columns:
    gflops = data_at_1200[kernel]
    relative_performance = (gflops / openblas_gflops) * 100
    markdown_table += f"| {kernel} | `{gflops:.1f}` | {relative_performance:.1f}% |\n"

markdown_table += "<!-- benchmark_results -->\n"

# Save the Markdown table
with open('./benchmark_results.md', 'w') as f:
    f.write(markdown_table)
