import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
from multiprocessing import freeze_support

def main():
# a) incarcarea setului de date
    df = pd.read_csv("Sesiune/BostonHousing.csv", dtype=float)

    # extrag din dataset variabilele care prezinta interes
    valoarea_locuintelor = df["medv"].values
    nr_camere = df["rm"].values
    rata_crim = df["crim"].values
    supraf_comerciala = df["indus"].values

    # ex a)
    #creez o matrice care contine variabilele independente (nr_camere, rata_crim, supraf_comerciala), 
    # iar matrice_mean reprezinta media acestor variabile
    matrice = np.column_stack((nr_camere, rata_crim, supraf_comerciala))
    matrice_mean = matrice.mean(axis=0, keepdims=True)

    #analizam datele
    print(f"-- Media variabilelor independente: {matrice_mean}")   
    print(f"-- Media variabilei dependente: {valoarea_locuintelor.mean()}")
    print(f"-- Deviatia standard: {matrice.std(axis=0, keepdims=True)}")
    print(f"-- Output deviatie standard: {valoarea_locuintelor.std()}")

    # b) model pymc
    with pm.Model() as model:
        # definirea variabilelor independente
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=1, shape=3)
        eps = pm.HalfCauchy('ϵ', 5000)
        ν = pm.Exponential('ν', 1/30)
        mat_distrib = pm.MutableData('mat_distrib', matrice)
        #  variabila determinista miu =  fct a coeficientilor si a datelor independente.
        miu = pm.Deterministic('miu',alpha + pm.math.dot(mat_distrib, beta)) 

        # definire distributia variabilei dependente
        medv = pm.Normal('medv', mu=miu, sigma=eps, observed=valoarea_locuintelor)

        idata_mlr = pm.sample(1250, return_inferencedata=True)

# c)
# vizualizam  distrib a posteriori a parametrilor
    az.plot_forest(idata_mlr, hdi_prob=0.95, var_names=['beta'])
    plt.show()
    print(az.summary(idata_mlr, hdi_prob=0.95, var_names=['beta']))

if __name__ == '__main__':
    freeze_support()
    main()