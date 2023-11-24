'''
Esti administratorul unui centru de service. Un aspect important pentru clienti este timpul mediu de asteptare 
la telefon, pe care vrei sa il estimezi. Propunem un model de inferenta bayesiana astfel:
timpul mediu de asteptare e modelat de o distributie normala de parametri niu si sigma. 
generati 200 de timpi medii de asteptare cu distriburia de verioimilitate cu parametri alesi de voi
descrieti modelul in pyMC, folosind distrib a posteriori pt parametrul sigma.
estimeaza cu ajutorul modelului de mai sus distributia a posteriori pt sigma.
'''
import random
import numpy as np
from scipy import stats
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az
from multiprocessing import freeze_support


def main():
    # generare 200 timpi medii de asteptare
    timp_mediu_asteptare = np.random.normal(loc=10, scale=2, size=200)

    with pm.Model() as model:
        # parametrii distributiei 
        # miu este normal distribuit cu media 0 si deviatia standard 10;
        mu = pm.Normal('mu', mu=0, sigma=10)
        # sigma este halfNormal ptc nu poate fi negativ; nu stim nimic despre el
        sigma = pm.HalfNormal('sigma', sigma=1)

        # observatii reprezinta distributia timpilor de asteptare
        observatii = pm.Normal('observatii', mu=mu, sigma=sigma, observed=timp_mediu_asteptare)

        trace = pm.sample(2000, tune=1000)

    az.plot_posterior(trace, var_names=['sigma'])

    plt.title('Distributia a posteriori pentru sigma')
    plt.xlabel('Sigma')
    plt.ylabel('Densitatea de probabilitate')
    plt.show()

if __name__ == '__main__':
    freeze_support()
    main()