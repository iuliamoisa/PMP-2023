import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from multiprocessing import freeze_support


def main():
# ex1.
# Modelul probabilistic

    with pm.Model() as model:
        # pt ex 1, pp alpha=1
        alpha = 1  # cu cat e mai mare, timpul de preg e mai scurt    
        traffic = pm.Poisson('T', mu=20)  # nr clienti care intra in locatie
        order_time = pm.Normal('O', mu=2, sigma=0.5)  # timpul de plasare & plata a comenzii
        cook = pm.Exponential("C", lam=alpha)
        trace = pm.sample(20000, chains=1)
        
    # ex2.
        
        # dictionary = {
        #                 'timp_asteptare': trace['O'].tolist(),
        #                 'timp_pregatire': trace['C'].tolist(),
        #             }
        # df = pd.DataFrame(dictionary)


        # timp_sub_15 = df[(df['timp_asteptare'] + df['timp_pregatire']<=15)]
        # size_sample = df.shape[0]
        # procentaj_sub_15 = timp_sub_15.shape[0]/size_sample
        # print("Procentajul timpului de asteptare sub 15 minute este:", procentaj_sub_15)
    az.plot_posterior(trace)
    plt.show()
if __name__ == '__main__':
    freeze_support()
    main()
