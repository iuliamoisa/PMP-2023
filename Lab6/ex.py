import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from multiprocessing import freeze_support

def main():
    Y_values = [0, 5, 10]
    theta_values = [0.2, 0.5]

    fig, axs = plt.subplots(len(Y_values), len(theta_values), figsize=(15, 15)) # creare figura cu subfiguri (axs)

    for i, Y in enumerate(Y_values):
        for j, theta in enumerate(theta_values):
            with pm.Model() as model:
                n = pm.Poisson('n', mu=10)
                y = pm.Binomial('y', n=n, p=theta, observed=Y) # verosimilitatea
                trace = pm.sample(2000, tune=1000)
                az.plot_posterior(trace, ax=axs[i, j])
                axs[i, j].set_title(f'Y = {Y}, θ = {theta}')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    freeze_support()
    main()