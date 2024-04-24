import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
## example: https://stackoverflow.com/questions/56985271/linear-congruential-generator-how-to-choose-seeds-and-statistical-tests ###
# RNG for exponential distribution
class CustomRandomGenerator:
    def __init__(self, seed=None):
        self.seed = seed
        self.a = 1664525
        self.c = 1013904223
        self.m = 2**32
        self.current = seed

    def rand(self):
        self.current = (self.a * self.current + self.c) % self.m
        return self.current / self.m

def custom_erlang(k, theta, rng):
    result = 0
    for _ in range(k):
        result += -theta * np.log(1 - rng.rand())
    return result

# Initialize the custom random number generator
rng = CustomRandomGenerator(seed=78727)

# Generate samples from Erlang distribution
sample_size = 1000
k = 2
theta = 3
sample = [custom_erlang(k, theta, rng) for _ in range(sample_size)]

# Plot histogram for Erlang distribution
plt.figure(figsize=(8, 6))
plt.hist(sample, bins=30, density=True, color='skyblue', edgecolor='black', alpha=0.7, label='Empirical Distribution')
plt.hist(stats.gamma.rvs(a=k, scale=theta, size=sample_size), bins=30, density=True, color='red', edgecolor='black', alpha=0.5, label='Theoretical PDF')
plt.title('Erlang Distribution Comparison')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.show()

# Perform hypothesis testing
# Kolmogorov-Smirnov test
ks_stat, ks_p = stats.kstest(sample, 'gamma', args=(k, 0, theta))

# Calculate mean for empirical distribution
mean_empirical = np.mean(sample)

# Calculate theoretical mean
mean_theoretical = k * theta

# Calculate shape and scale parameters for theoretical distribution
shape_theoretical, _, scale_theoretical = stats.gamma.fit(sample, floc=0)

# Define data for the table
data = {
    'SHAPE (K)': [shape_theoretical, k],
    'SCALE (THETA)': [scale_theoretical, theta]
}

# Create DataFrame
df = pd.DataFrame(data, index=['EMPIRICAL', 'THEORETICAL'])
print(df)

# Print test results
alpha = 0.05
if ks_p < alpha:
    print('Kolmogorov-Smirnov test: statistic = {}, p-value = {}'.format(ks_stat, ks_p))
    print('Reject the null hypothesis')
else:
    print('Kolmogorov-Smirnov test: statistic = {}, p-value = {}'.format(ks_stat, ks_p))
    print('Fail to reject the null hypothesis')


observed_freq, _ = np.histogram(sample, bins=30, density=True)
chi2_stat_1, chi2_p_1 = stats.chisquare(observed_freq)
alpha = 0.05
if chi2_p_1 < alpha:
    print('Chi-square test: statistic = {}, p-value = {}'.format(chi2_stat_1, chi2_p_1))
    print('Reject the null hypothesis')
else:
    print('Chi-square test: statistic = {}, p-value = {}'.format(chi2_stat_1, chi2_p_1))
    print('Fail to reject the null hypothesis')
