import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

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

def custom_normal(mean, std_dev, rng):
    u1 = rng.rand()
    u2 = rng.rand()
    z = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    return mean + std_dev * z

# Initialize the custom random number generator
rng = CustomRandomGenerator(seed=42)

# Generate samples from Normal distribution
sample_size = 1000
mean = 2
std_dev = 0.2
sample = [custom_normal(mean, std_dev, rng) for _ in range(sample_size)]

# Plot histogram for normal distribution
plt.figure(figsize=(8, 6))
plt.hist(sample, bins=30, density=True, color='skyblue', edgecolor='black', alpha=0.7, label='Empirical Distribution')
plt.hist(np.random.normal(mean, std_dev, size=sample_size), bins=30, density=True, color='red', edgecolor='black', alpha=0.5, label='Theoretical PDF')
plt.title('Normal Distribution Comparison')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.show()

# Calculate mean and standard deviation for empirical distribution
mean_empirical = np.mean(sample)
std_dev_empirical = np.std(sample)

# Calculate theoretical mean and standard deviation
mean_theoretical = mean
std_dev_theoretical = std_dev

# Create DataFrame for the table
data = {
    'MEAN': [mean_empirical, mean_theoretical],
    'STANDARD DEVIATION': [std_dev_empirical, std_dev_theoretical]
}

df = pd.DataFrame(data, index=['EMPIRICAL', 'THEORETICAL'])
print(df)

# Perform hypothesis testing
# Kolmogorov-Smirnov test
ks_stat, ks_p = stats.kstest(sample, 'norm', args=(mean, std_dev))
alpha = 0.05
if ks_p < alpha:
    print('Kolmogorov-Smirnov test: statistic = {}, p-value = {}'.format(ks_stat, ks_p))
    print('Reject the null hypothesis')
else:
    print('Kolmogorov-Smirnov test: statistic = {}, p-value = {}'.format(ks_stat, ks_p))
    print('Fail to reject the null hypothesis')

# Chi-square test
observed_freq, _ = np.histogram(sample, bins=30, density=True)
chi2_stat_1, chi2_p_1 = stats.chisquare(observed_freq)
alpha = 0.05
if chi2_p_1 < alpha:
    print('Chi-square test: statistic = {}, p-value = {}'.format(chi2_stat_1, chi2_p_1))
    print('Reject the null hypothesis')
else:
    print('Chi-square test: statistic = {}, p-value = {}'.format(chi2_stat_1, chi2_p_1))
    print('Fail to reject the null hypothesis')


