import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Read the data from the file
N_values = []
gflops_values = []
with open('gflops_results.txt', 'r') as file:
    for line in file:
        N, gflops = line.split(',')
        N_values.append(int(N))
        gflops_values.append(float(gflops))

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(N_values, gflops_values, marker='x', markersize=5, linewidth=2, label="GFLOPS vs N")
plt.title('GFLOPS vs N')
plt.xlabel('N')
plt.ylabel('GFLOPS')
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig("gflops_vs_N.png", dpi=300)  # High quality

# Display the plot
plt.show()
