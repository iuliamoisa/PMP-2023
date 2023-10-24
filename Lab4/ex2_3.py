import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from multiprocessing import freeze_support


def main():
# ex1. modelul probabilistic
    with pm.Model() as model:  
        # pt ex 1, pp alpha=1
        alpha = 1  # cu cat e mai mare, timpul de preg e mai scurt    
        traffic = pm.Poisson('T', mu=20)  # nr clienti care intra in locatie
        order_time = pm.Normal('O', mu=2, sigma=0.5)  # timpul de plasare & plata a comenzii
        cook = pm.Exponential("C", lam=alpha)
        trace = pm.sample(500)
    az.plot_posterior(trace)
    plt.show()

#ex2.

    # calc probab de a servi comenzile < 15min pt un alfa dat
    def servire_sub_15(alpha):
        total_time = np.random.normal(loc=2, scale=0.5, size=1000) + np.random.exponential(scale=1/alpha, size=1000)
        num_success = np.sum(total_time < 15)  # nr comenzi servite in mai putin de 15min
        success_prob = num_success / 1000  # 1000 simulari=> # caz fav/ # cazuri posibile
        return success_prob

    # estimez alfa
    alpha_values = np.linspace(0.01, 1, 100)  #  def 100 de valori ale parametrului alfa intre 0.01 si 1
    success_probabilities = []

    # pt fiecare alfa calc probabilitatea de a servi <15min
    for alpha in alpha_values:
        success_prob = servire_sub_15(alpha)
        success_probabilities.append(success_prob)

    success_probabilities = np.array(success_probabilities)
    alpha_max_index = np.argmax(success_probabilities)
    alpha_max = alpha_values[alpha_max_index]

    print('alfa maxim = ', alpha_max)


#ex 3. timpul mediu de astept pt un client

    rate_of_arrival = 20  #  nr clienti/h
    occupancy_rate = alpha_max  #  timp de pregatire a comenzii = cat timp statia de gatit nu poate lua comenzi noi
    waiting_time = (1 / rate_of_arrival) * (1 / (1 - occupancy_rate)) # rata de asteptare la coada

    print(f'Timpul mediu de asteptare pentru a fi servit unui client: {waiting_time * 60:.2f} minute')

if __name__ == '__main__':
    freeze_support()
    main()