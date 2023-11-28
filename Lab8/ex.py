import pandas as pd
import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from multiprocessing import freeze_support
def main():
    # Încărcăm datele
    data = pd.read_csv('Lab8/Prices.csv')

    # Definim variabilele
    price = data['Price']
    speed = data['Speed']
    hard_drive = np.log(data['HardDrive'])

    with pm.Model() as model:
        # Definim distribuțiile a priori
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta1 = pm.Normal('beta1', mu=0, sigma=10)
        beta2 = pm.Normal('beta2', mu=0, sigma=10)
        sigma = pm.HalfNormal('sigma', sigma=1)

        # Definim media
        mu = alpha + beta1*speed + beta2*hard_drive

        # Definim distribuția a posteriori
        y = pm.Normal('y', mu=mu, sigma=sigma, observed=price)


        idata = pm.sample(2000, tune=2000, return_inferencedata=True) # sample-ul, adica distributia a posteriori 

    az.plot_trace(idata, var_names=['alpha', 'beta1', 'beta2', 'sigma'])

    plt.show()

if __name__ == '__main__':
    freeze_support()
    main()

