import pymc as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from multiprocessing import freeze_support

def main():
    data = pd.read_csv('Lab9/Admission.csv')

    GRE = data['GRE'].values
    GPA = data['GPA'].values
    Admission = data['Admission'].values

    plt.scatter(GRE[Admission == 0], GPA[Admission == 0], color='red', label='Picat')
    plt.scatter(GRE[Admission == 1], GPA[Admission == 1], color='green', label='Admis')

    plt.xlabel('GRE')
    plt.ylabel('GPA')
    plt.legend()
    plt.show()

    # normalizare date de intrare
    gre_min = np.min(GRE)
    gre_max = np.max(GRE)
    norm_gre = (GRE - gre_min) / (gre_max - gre_min) 

    gpa_min = np.min(GPA)
    gpa_max = np.max(GPA)
    norm_gpa = (GPA - gpa_min) / (gpa_max - gpa_min)


    # constr modelului logistic
    with pm.Model() as m:
        alfa = pm.Normal('alfa', mu=0, sigma=10)
        beta1 = pm.Normal('beta1', mu=0, sigma=2)
        beta2 = pm.Normal('beta2', mu=0, sigma=2)

        miu = alfa + beta1 * norm_gre + beta2 * norm_gpa
        theta = pm.Deterministic('theta', pm.math.sigmoid(miu))
        
        # granita de decizie
        granita = pm.Deterministic('granita', -alfa/beta2 - beta1/beta2 * norm_gre) 

        adm = pm.Bernoulli('adm', p=theta, observed=Admission)

        trace = pm.sample(2000, return_inferencedata=True)

    # vizualizare granita de decizie si intervalul HDI in jurul ei
    alfa_m = trace['posterior']['alfa'].mean().item()
    beta1_m = trace['posterior']['beta1'].mean().item()
    beta2_m = trace['posterior']['beta2'].mean().item()

    plt.scatter(norm_gre, norm_gpa, c=[f"C{x}" for x in Admission]) 
    plt.xlabel('gre_norm')
    plt.ylabel('gpa_norm')

    plt.plot(norm_gre, -alfa_m/beta2_m - beta1_m/beta2_m * norm_gre, c='k')
    az.plot_hdi(norm_gre, trace['posterior']['granita'], hdi_prob=0.94, color='k')

    plt.show()

    alfa_p = trace['posterior']['alfa']
    beta1_p = trace['posterior']['beta1']
    beta2_p = trace['posterior']['beta2']
    

    # 3 si 4

    new_gre_1 = (550 - gre_min) / (gre_max - gre_min)
    new_gpa_1 = (3.5 - gpa_min) / (gpa_max - gpa_min)
    new_prob_1 = alfa_p + beta1_p * new_gre_1 + beta2_p * new_gpa_1

    stacked_1 = az.extract(new_prob_1)
    new_prob_values_1 = stacked_1.x.values
    new_prob_values_1 = 1 / (1 + np.exp(-new_prob_values_1))

    hdi_prob_1 = az.hdi(new_prob_values_1, hdi_prob=0.9)
    print("Intervalul HDI pentru probabilitatea de admitere (scor GRE 550, GPA 3.5):", hdi_prob_1)

    ########## 
    
    new_gre_2 = (500 - gre_min) / (gre_max - gre_min)
    new_gpa_2 = (3.2 - gpa_min) / (gpa_max - gpa_min)
    new_prob_2 = alfa_p + beta1_p * new_gre_2 + beta2_p * new_gpa_2

    stacked_2 = az.extract(new_prob_2)
    new_prob_values_2 = stacked_2.x.values
    new_prob_values_2 = 1 / (1 + np.exp(-new_prob_values_2))

    hdi_prob_2 = az.hdi(new_prob_values_2, hdi_prob=0.9)
    print("Intervalul HDI pentru probabilitatea de admitere (scor GRE 500, GPA 3.2):", hdi_prob_2)


if __name__ == '__main__':
    freeze_support()
    main()
