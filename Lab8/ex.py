import pandas as pd
import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from multiprocessing import freeze_support

def main():
    data = pd.read_csv('Lab8/Prices.csv')
# def variabile
    price = data['Price']
    speed = data['Speed']
    hard_drive = np.log(data['HardDrive'])
    prem = data['Premium_Bonus'] = (data['Premium'] == 'yes').astype(int)

    # ex. 1)
    with pm.Model() as model: 
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta1 = pm.Normal('beta1', mu=0, sigma=10)
        beta2 = pm.Normal('beta2', mu=0, sigma=10)
        beta_premium = pm.Normal('beta_premium', mu=0, sigma=10) 
        sigma = pm.HalfCauchy('sigma', 5)

        mu = pm.Deterministic('mu', alpha + beta1 * speed + beta2 * hard_drive + beta_premium * prem)

        y = pm.Normal('y', mu=mu, sigma=sigma, observed=price)
        trace = pm.sample(2000, tune=2000, return_inferencedata=True)

    # ex. 2)
    hdi_beta1 = pm.stats.hdi(trace['beta1'], hdi_prob=0.95)
    hdi_beta2 = pm.stats.hdi(trace['beta2'], hdi_prob=0.95)
    print("Estimarea HDI pentru beta1 si beta 2 sunt: ", hdi_beta1 , hdi_beta2)

    hdi_beta_premium = pm.stats.hdi(trace['beta_premium'], hdi_prob=0.95)
    print("Estimarea HDI pentru beta_premium:", hdi_beta_premium)


    ''' ex. 3)
    beta1 si beta2 corespund coeficientilor asociati frecventei procesorului si dimensiunii hard disk-ului
    intervalele HDI sunt calculate a.i sa acopere cu 95% cele mai probabile valori pentru acesti coeficienti
    intervalele HDI pentru beta1 si beta2 nu includ 0 => 
        frecventa procesorului si dimensiunea hard disk-ului sunt considerate predictori utili ai 
        pretului de vanzare al PC-urilor. adica acesti doi factori au o influenta semnificativa asupra 
        determinarii preturilor de vanzare.
    '''

    # ex. 5)
    predictive_samples = pm.sample_posterior_predictive(trace, samples=5000, model=model)['y'] 
    specific_processor_freq = 33
    specific_hard_disk_size = 540
    specific_simulated_prices = pm.sample_posterior_predictive(trace, samples=5000, model=model, var_names=['mu'])['mu']

    specific_prices = specific_simulated_prices[:, (speed == specific_processor_freq) & (hard_drive == np.log(specific_hard_disk_size))]

    # ex. 4)
    specific_hdi_price = pm.stats.hdi(specific_prices, hdi_prob=0.90)
    print("Estimarea HDI pentru pretul de vanzare asteptat:", specific_hdi_price) 

    if 0 not in hdi_beta_premium:
        print("Faptul ca producatorul este premium afecteaza semnificativ pretul.")
    else:
        print("Nu exista suficiente care sa ateste ca daca producatorul este premium pretul este afectat")
    '''
    intervalul obtinut pt beta_premium este larg si poate include 0
    => atributul 'Premium' nu are mereu un impact semnificativ asupra predictiei pretului
    => un producator premium nu contribuie in mod constant la variatia preturilor PC-urilor
    '''

if __name__ == '__main__':
    freeze_support()
    main()

