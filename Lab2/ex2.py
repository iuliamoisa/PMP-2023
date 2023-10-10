import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)

z = [] # timp total servire a fiecarui client
nr = 0 # cate cazuri au timp de servire > 3 milisec

for _ in range(10000): # simulez 1000 clienti
    y = np.random.randint(1, 101) # decid random catre care server redirectionez clientul
    latenta = stats.expon(scale=1/0.25).rvs()
    if y <= 25:
        m = stats.gamma.rvs(4, 0, 1/3) # generez timpul de servire cu gamma specificat pt serv1 
    elif 50 >= y > 25:
        m = stats.gamma.rvs(4, 0, 1/2)
    elif 50 < y <= 80:
        m = stats.gamma.rvs(5, 0, 1/2)
    else:
        m = stats.gamma.rvs(5, 0, 1/3)
    z.append(m + latenta)

    if m + latenta >= 3: # verif daca timpul total > 3mili
        nr += 1

probabilitate = nr / 10000
print("Probabilitatea ca timpul sa fie peste 3 este:", probabilitate)


az.plot_posterior({'z':z})
plt.show() 
