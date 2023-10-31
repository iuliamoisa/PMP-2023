import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import freeze_support

data = pd.read_csv('Lab5/trafic.csv')
traffic_counts = data['nr. masini'].values
count_data = np.loadtxt("trafic.csv", delimiter=',', dtype=int)
# Orele de (des)crestere a traficului
increased_times = [7*60, 16*60]
decreased_times = [8*60, 19*60]


def main():
# ex. 1 -- Modelul probabilistic
    with pm.Model() as traffic_model:
        # var iau valori intre 0 si val max a traficului
        lambda_ = pm.Uniform('lambda_', lower=0, upper=10) # upper = traffic_counts.max()
        lambda_increased = pm.Uniform('lambda_increased', lower=0, upper=15)
        lambda_decreased = pm.Uniform('lambda_decreased', lower=0, upper=7)

        # distrib poisson pt trafic normal:
        traffic_normal = pm.Poisson('traffic_normal', mu=lambda_, observed=traffic_counts)

        # distrib poisson pt trafic la orele 7, 16
        for time in increased_times:
            pm.Poisson(f'traffic_increased_{time}', mu=lambda_increased, observed=traffic_counts[data['minut'] == time])

        # distrib poisson pt trafic la orele 8, 19
        for time in decreased_times:
            pm.Poisson(f'traffic_decreased_{time}', mu=lambda_decreased, observed=traffic_counts[data['minut'] == time])

        # trace = pm.sample(500)
        # az.plot_posterior(trace)
        # plt.show()

        # ex 2:
        alpha = 1.0/count_data[:, 1].mean()

        lambda1 = pm.Exponential("l1", alpha)
        lambda2 = pm.Exponential("l2", alpha)
        lambda3 = pm.Exponential("l3", alpha)
        lambda4 = pm.Exponential("l4", alpha)
        lambda5 = pm.Exponential("l5", alpha)

        i1 = pm.Normal("i1", 60*3) # intervalele
        i2 = pm.Normal("i2", 60*11)
        i3 = pm.Normal("i3", 60*15)
        i4 = pm.Normal("i4", 60*20)

        #indicii pt momentele de schimbare
        tau1 = pm.DiscreteUniform("tau1", lower=1, upper=i1)
        tau2 = pm.DiscreteUniform("tau2", lower=tau1, upper=i2)
        tau3 = pm.DiscreteUniform("tau3", lower=tau2, upper=i3)
        tau4 = pm.DiscreteUniform("tau4", lower=tau3, upper=i4)

        lambda_ = pm.math.switch(pm.math.ge(data['minut'].values, tau4), lambda5,
                             pm.math.switch(pm.math.ge(data['minut'].values, tau3), lambda4,
                                           pm.math.switch(pm.math.ge(data['minut'].values, tau2), lambda3,
                                                         pm.math.switch(pm.math.ge(data['minut'].values, tau1), lambda2, lambda1))))
        
        observation = pm.Poisson('observation', lambda_, observed=traffic_counts)
        trace = pm.sample(2000)
    az.plot_posterior(trace)
    plt.show()
if __name__ == '__main__':
    freeze_support()
    main()