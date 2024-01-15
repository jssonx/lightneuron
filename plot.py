import matplotlib.pyplot as plt

# Data
version_implementations = [
    "naive",
    "naive+interchange loops",
    "naive+interchange loops+optimization flags",
    "naive+interchange loops+optimization flags+parallel loops",
    "naive+interchange loops+optimization flags+parallel tiling",
    "naive+interchange loops+optimization flags+parallel divide-and-conquer",
    "naive+interchange loops+optimization flags+parallel divide-and-conquer+avx2 intrinsics+data alignment",
    "naive+interchange loops+optimization flags+parallel tiling+avx2 intrinsics+data alignment",
    "naive+interchange loops+optimization flags+parallel divide-and-conquer+avx2 intrinsics+data alignment+coarsening",
    "Intel MKL"
]

percent_of_intel_mkl = [0.25, 0.65, 4.08, 18.62, 27.19, 30.76, 38.73, 44.13, 63.14, 100.00]

# Simplifying the version implementation names for the x-axis
simplified_versions = ["V" + str(i) for i in range(1, len(version_implementations) + 1)]

# Plotting with simplified labels
plt.figure(figsize=(10, 6))
plt.plot(simplified_versions, percent_of_intel_mkl, marker='o')
plt.xlabel('Version Implementation')
plt.ylabel('Percent of Intel MKL')
plt.title('Performance Comparison of Various Implementations vs Intel MKL')
plt.grid(True)
plt.tight_layout()

plt.savefig("./img/benchmark.png")

plt.show()
