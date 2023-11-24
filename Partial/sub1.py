''' un joc cu 2 jucatori: p0 si p1 astfel: 
se arunca mai intai o moneda normala pentru a decide cine incepe (p0 sau p1); 
in prima runda jucatorul desemnat arunca cu propria moneda: 
    se poate obtine o stema sau niciuna (adica n=0 sau 1, n=nr steme)!!!
in a 2 a runda celalalt jucator arunca cu moneda proprie de n+1 ori; 
        => m numarul de steme obtinute. 

jucatorul din prima runda castiga daca n>=m, altfel castiga jucatorul din runda 2.
p0 este necinstit, el aducand o moneda masluita, cu probabilitate de obtinere a stemei egala cu 1/3. moneda jucatorului p1 e normala. 
1)estimati care din cei 2 jucatori au sansele mai mari de castig, simuland un joc de 20000 ori
2) fol pgmpy def o retea bayesiana care sa descrie contextul de mai sus
3) fol modelul de mai sus determina ce fata a monedei e mai probabil sa se fi obtinut in prima runda, 
stiind ca in a doua nu s a obtinut nicio stema
'''

import random
from scipy import stats
from pgmpy.models import BayesianNetwork, BayesianModel
from pgmpy.factors.discrete import TabularCPD
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.inference import VariableElimination

jucator0_castiga = 0  # Variabila pentru numărul de jocuri câștigate de jucatorul 0
jucator1_castiga = 0  # Variabila pentru numărul de jocuri câștigate de jucatorul 1

for i in range(20000):  # Se simulează 20000 de jocuri
    p0 = 0
    p1 = 0
    moneda = random.random()  # arunc moneda si decid random cine incepe
    if moneda < 0.5:
        p0 = 1  # val<0,5 -> incepe primul
    else:
        p1 = 1  # altfel -> incepe al doilea

    
    if p0 == 1: # daca incepe primul, moneda lui e masluita-> sansa sa obtina stema e 1/3
        stema_moneda1 = stats.binom.rvs(1, 1 / 3)
    elif p1 == 1: # daca incepe al doilea
        stema_moneda1 = stats.binom.rvs(1, 0.5) # moneda nemasluita -> sansa normala sa obtina stema=1/2
    

    if stema_moneda1 == 1:
        n = 1  # daca a iesit stema, nr de steme=1
    else:
        n = 0

# in a  doua runda
    m = 0
    if p0 == 1: # a inceput primul, deci acum urmeaza al doilea 
        stema_moneda2 = stats.binom.rvs(1, 0.5,
                                        size=n + 1)
    elif p1 == 1: # a inceput al doilea, acum e randul primului jucator
         stema_moneda2 = stats.binom.rvs(1, 1 / 3,
                                        size=n + 1)

    for i in range(n + 1):
        if stema_moneda2[i] == 1:
            m += 1  # calculez nr de steme obtinute 

    if n >= m:  # n>=m => castiga jucatorul din prima runda; altfel castiga celalalt jucator, dina  doua runda
        if p0 == 1:
            jucator0_castiga += 1  
        else: jucator1_castiga += 1
    else:
        if p0 == 1:
            jucator1_castiga += 1  
        else: jucator0_castiga += 1

# calculez procentul de jocuri pe care le-a castigat fiecare; cazuri fav/cazuri posibile
print("Jucator P0 ---> ", jucator0_castiga / 20000 * 100, "%")
print("Jucator P1 --->", jucator1_castiga / 20000 * 100, "%")

if jucator0_castiga / 20000 * 100 >  jucator1_castiga / 20000 * 100:
    print('P0 are sanse mai mari de castig')
else: print('P1 are sanse mai mari de castig')



# Modelul Bayesian
model = BayesianNetwork([('Prim', 'n'), ('n', 'm'), ('Prim', 'm')])

# jucatorul care incepe -> 2 posibilitati, fiecare probabilitate 1/2
primul = TabularCPD('Prim', 2, [[0.5], [0.5]])

# definesc distributiile  conditionate 
cpd_n = TabularCPD('n', 2, [[1/3, 0.5], [2/3, 0.5]], evidence=['Prim'], evidence_card=[2]) # 

cpd_m = TabularCPD('m', 2, [[1/3, 2/3, 0.5, 0.5], [2/3, 1/3, 0.5, 0.5]], evidence=['n', 'Prim'], evidence_card=[2, 2])

model.add_cpds(primul, cpd_n, cpd_m)

# Verificarea modelului
model.check_model()

# Desenarea modelului
pos = nx.circular_layout(model)
nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
plt.show()

# inferenta, adica calcularea probabilitatilor conditionate; 
from pgmpy.inference import VariableElimination


infer = VariableElimination(model) 
# inferenta este facuta cu ajutorul algoritmului VariableElimination

# calculez probabilitatea ca in prima runda sa fi iesit stema, stiind ca in a doua nu a iesit nicio stema
prob_n = infer.query(variables=['n'], evidence={'m': 0}) # stema nu a fost obtinuta in a doua runda
print(prob_n)



