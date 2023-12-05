import pymc as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from multiprocessing import freeze_support

def logistic_function(x, beta):
    return 1 / (1 + np.exp(-(beta[0] + beta[1] * x)))

def main():
    data = pd.read_csv('Lab9/Admission.csv')

    GRE = data['GRE'].values
    GPA = data['GPA'].values
    Admission = data['Admission'].values

    with pm.Model() as logistic_model:
        beta0 = pm.Normal('beta0', mu=0, sigma=10)
        beta1 = pm.Normal('beta1', mu=0, sigma=10)
        beta2 = pm.Normal('beta2', mu=0, sigma=10)

        pi = pm.Deterministic('pi', pm.math.invlogit(beta0 + beta1 * GRE + beta2 * GPA))

        admission_observed = pm.Bernoulli('admission_observed', p=pi, observed=Admission)

        trace = pm.sample(2000, tune=1000, target_accept=0.9)

    print(pm.summary(trace).round(2))
    beta_samples = trace['beta']
    beta_means = beta_samples.mean(axis=0)
    beta_hdi = az.hdi(beta_samples, hdi_prob=0.94)

if __name__ == '__main__':
    freeze_support()
    main()