import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pymc as pm
import pytensor.tensor as pt
from sklearn.mixture import GaussianMixture

if __name__ == '__main__':
    # ex1: generare 500 de date dintr-o mixtura de 3 distrib Gaussiene 
    # cu medii de 5 (200 date), 0(150 date), -3(150 date) si dev standard de 2, 2, 1.

    clusters = 3
    n_cluster = [200, 150, 150]
    n_total = sum(n_cluster)
    means = [5, 0, -3]
    std_devs = [2, 2, 1]
    mix = np.random.normal(np.repeat(means, n_cluster),
    np.repeat(std_devs, n_cluster))
    #print(mix)
    cs_exp = np.array(mix)
    az.plot_kde(np.array(mix))
    plt.show()

    #ex2: calibrare
    clusters = [2, 3, 4]
    models = []
    idatas = []
    

    for cluster in clusters:
        with pm.Model() as model:
            p = pm.Dirichlet('p', a=np.ones(cluster))
            means = pm.Normal('means',
                              mu=np.linspace(cs_exp.min(), cs_exp.max(), cluster),
                              sigma=10, shape=cluster,
                              transform=pm.distributions.transforms.ordered)
            sd = pm.HalfNormal('sd', sigma=10)
            y = pm.NormalMixture('y', w=p, mu = means, sigma=sd, observed=cs_exp)
            idata = pm.sample(100, tune=100, target_accept=0.9, random_seed=10, return_inferencedata=True)
            idatas.append(idata)
            models.append(model)

    _, ax = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    ax = np.ravel(ax)
    x = np.linspace(cs_exp.min(), cs_exp.max(), 200)

    #ex 3
    