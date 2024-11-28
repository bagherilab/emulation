import numpy as np
import matplotlib.pyplot as plt

# Generate observed data
np.random.seed(42)
true_mu = 5.0  # True mean
sigma = 1.0    # Known standard deviation
N = 20         # Number of observations
y_obs = np.random.normal(loc=true_mu, scale=sigma, size=N)

# ABC Parameters
mu_prior_mean = 0.0     # Prior mean
mu_prior_std = 3.0      # Prior standard deviation
epsilon = 30.0           # Tolerance threshold
num_samples = 100000     # Number of prior samples

# ABC Sampling
accepted_samples = []
for _ in range(num_samples):
    # Sample from prior
    mu = np.random.normal(mu_prior_mean, mu_prior_std)
    
    # Simulate data
    y_sim = np.random.normal(mu, sigma, size=N)
    
    # Compute distance
    distance = np.sum((y_sim - y_obs)**2)
    # Accept or reject
    if distance < epsilon:
        accepted_samples.append(mu)

# Plot posterior
plt.figure(figsize=(10, 6))
plt.hist(accepted_samples, bins=30, density=True, alpha=0.6, color="blue", label="Posterior (ABC)")
plt.axvline(true_mu, color="red", linestyle="--", label="True Mean")
plt.title("Posterior Distribution of $\mu$ using ABC")
plt.xlabel("$\mu$")
plt.ylabel("Density")
plt.legend()
plt.grid()
plt.savefig("abc_posterior.png")
