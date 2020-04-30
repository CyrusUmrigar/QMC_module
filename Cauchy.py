import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

n_bins = 500 #number of bins in histogram
n_var = 16 #number of random variables to sum over

def sum_rand_var(n_samples, n_var, distrib):
    samples = [sum(distrib(n_var))/n_var for n in range(n_samples)]
    return samples

g1 = np.random.randn(800000)
plt.hist(g1, bins=n_bins, range=[-8, 8], density=True, histtype='step', color='red')

g2 = sum_rand_var(80000, n_var, np.random.randn)
plt.hist(g2, bins=n_bins, range=[-8, 8], density=True, histtype='step', color='green')

c1 = np.random.standard_cauchy(800000)
plt.hist(c1, bins=n_bins, range=[-8, 8], density=True, histtype='step', color='blue')

c2 = sum_rand_var(80000, n_var, np.random.standard_cauchy)
plt.hist(c2, bins=n_bins, range=[-8, 8], density=True, histtype='step', color='magenta')

plt.title("Demonstration that Cauchy distribution is invariant under averaging")
custom_lines = [Line2D([0], [0], color = 'red', lw=2),
                Line2D([0], [0], color = 'green', lw=2),
                Line2D([0], [0], color = 'blue', lw=2),
                Line2D([0], [0], color = 'magenta', lw=2)]
plt.legend(custom_lines, ['Normal', 'Ave of 16 Normal', 'Cauchy', 'Ave of 16 Cauchy'])

plt.savefig('Cauchy.pdf')

plt.show()
