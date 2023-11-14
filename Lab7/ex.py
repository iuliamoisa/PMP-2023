import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm
import arviz as az
import numpy as np

if __name__ == '__main__':
    # incarc datele + elim valorile lipsa
    df = pd.read_csv('Lab7/auto-mpg.csv')
    df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
    df = df.dropna(subset=['horsepower', 'mpg'])

    # ------ a) Vizualizarea relatiei dintre CP și mpg:
    plt.figure(figsize=(10, 6))
    plt.scatter(df['horsepower'], df['mpg'])
    plt.xlabel('Cai putere (CP)')
    plt.ylabel('Mile pe galon (mpg)')
    plt.title('Relația dintre CP și mpg')
    plt.show()

    CP = df['horsepower'].values
    mpg = df['mpg'].values
    # --- b) modelul in pymc
    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0, sd=10) # intercept
        beta = pm.Normal('beta', mu=0, sd=10) # coef de reg
        mu = alpha + beta * df['horsepower']
        sigma = pm.HalfNormal('sigma', sd=1)
        mpg = pm.Normal('mpg', mu=mu, sd=sigma, observed=df['mpg']) # verosimilitate

    # --- c) determin dreapta de regresie cu MAP
    with model:
        map_estimate = pm.find_MAP() # gasesc estim a posteriori max aka MAP
        alpha_map = map_estimate['alpha'] # extrag estimarile pt intercept si coef 
        beta_map = map_estimate['beta']
        print(f'Dreapta de regresie: y = {alpha_map:.2f} + {beta_map:.2f} * CP')


    # Creez date pentru linia de regresie (100 val uniform distrib intre min si max CP)
    x_values = np.linspace(min(df['horsepower']), max(df['horsepower']), 100)
    y_values = alpha_map + beta_map * x_values

    plt.scatter(df['horsepower'], df['mpg'])
    plt.plot(x_values, y_values, color='red')
    plt.title('Regresia liniara')
    plt.xlabel('Cai Putere (CP)')
    plt.ylabel('Mile pe Galon (mpg)')
    plt.show()

    # d) 
    # cu cat regiunea 95% HDI e mai ingusta, cu atat modelul e mai sigur in predictii
    #daca este mai mare, incertitudinea modelului creste
    # with model:
    #     trace = pm.sample(5000, tune=1000)

    # x_values = np.linspace(min(df['horsepower']), max(df['horsepower']), 100)
    # y_values = alpha_map + beta_map * x_values

    # plt.scatter(df['horsepower'], df['mpg'])
    # plt.plot(x_values, y_values, color='red', label='Dreapta de regresie')
    # az.plot_hdi(x_values, trace['mpg'], color='blue', fill_kwargs={'alpha': 0.2}, label='95% HDI')
    # plt.title('Regresia liniara cu HDI')
    # plt.xlabel('Cai Putere (CP)')
    # plt.ylabel('Mile pe Galon (mpg)')
    # plt.legend()
    # plt.show()
