import arviz as az
import matplotlib.pyplot as plt
import numpy as np 

centered_eight_data = az.load_arviz_data("centered_eight")
non_centered_eight_data = az.load_arviz_data("non_centered_eight")

print("Modelul centrat:")
print("Nr de lanturi:", centered_eight_data.posterior.chain.size)
print("Marimea totala a esantionului generat:", centered_eight_data.posterior.draw.size)

az.plot_posterior(centered_eight_data)

print("\nModelul necentrat:")
print("Nr de lanțuri:", non_centered_eight_data.posterior.chain.size)
print("Marimea totala a esantionului generat:", non_centered_eight_data.posterior.draw.size)

az.plot_posterior(non_centered_eight_data)

plt.show()


################## 

rhats_centered = az.rhat(centered_eight_data, var_names=['mu', 'tau'])
autocorrelation_mu_centered = az.autocorr(centered_eight_data.posterior["mu"].values)
autocorrelation_tau_centered = az.autocorr(centered_eight_data.posterior["tau"].values)

rhats_non_centered = az.rhat(non_centered_eight_data, var_names=['mu', 'tau'])
autocorrelation_mu_non_centered = az.autocorr(non_centered_eight_data.posterior["mu"].values)
autocorrelation_tau_non_centered = az.autocorr(non_centered_eight_data.posterior["tau"].values)

data = np.array([
    ["Centrat", rhats_centered['mu'].item(), rhats_centered['tau'].item(),
     autocorrelation_mu_centered.mean().item(), autocorrelation_tau_centered.mean().item()],
    ["Necentrat", rhats_non_centered['mu'].item(), rhats_non_centered['tau'].item(),
     autocorrelation_mu_non_centered.mean().item(), autocorrelation_tau_non_centered.mean().item()]
])

header = ["Model", "Rhat_mu", "Rhat_tau", "Autocorrelation_mu", "Autocorrelation_tau"]
table = np.vstack([header, data])
print(table)

# obs ca modelul necentrat are autocorelatia mai scazuta adica va avea o convergenta mai rapida,
# deoarece valorile generate sunt mai independente si nu depind atat de mult de valorile anterioare

################# 

divergences_centered = centered_eight_data.sample_stats.diverging.sum()
divergences_non_centered = non_centered_eight_data.sample_stats.diverging.sum()

print(f'Nr de divergente pentru modelul centrat: {divergences_centered}')
print(f'Nr de divergente pentru modelul necentrat: {divergences_non_centered}')

az.plot_pair(centered_eight_data, var_names=['mu', 'tau'], divergences=True, kind='scatter')
plt.title("Divergente (pt centrat)")
plt.show()

az.plot_pair(non_centered_eight_data, var_names=['mu', 'tau'], divergences=True, kind='scatter')
plt.title("Divergente (pt necentrat)")
plt.show()
