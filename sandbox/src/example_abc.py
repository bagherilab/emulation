import numpy as np
import matplotlib.pyplot as plt

# Generate observed data
np.random.seed(42)
true_mu = 5.0  # True mean
sigma = 1.0    # Known standard deviation
N = 30         # Number of observations
y_obs = np.random.normal(loc=true_mu, scale=sigma, size=N)

# ABC Parameters
mu_prior_mean = 0.0     # Prior mean
mu_prior_std = 3.0      # Prior standard deviation
epsilon = 50.0           # Tolerance threshold
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
fig, axs = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

# Plot prior distribution
mu_prior_samples = np.random.normal(mu_prior_mean, mu_prior_std, num_samples)
axs[0].hist(mu_prior_samples, bins=30, density=True, alpha=0.6, color="green", label="Prior")
axs[0].axvline(true_mu, color="red", linestyle="--", label="True Mean")
axs[0].set_title("Prior Distribution of $\mu$")
axs[0].set_xlabel("$\mu$")
axs[0].set_ylabel("Density")
axs[0].legend()
axs[0].grid()

# Plot posterior distribution
axs[1].hist(accepted_samples, bins=30, density=True, alpha=0.6, color="blue", label="Posterior (ABC)")
axs[1].axvline(true_mu, color="red", linestyle="--", label="True Mean")
axs[1].set_title("Posterior Distribution of $\mu$ using ABC")
axs[1].set_xlabel("$\mu$")
axs[1].set_ylabel("Density")
axs[1].legend()
axs[1].grid()

plt.tight_layout()
plt.savefig("abc_prior_posterior.png")
