import numpy as np
from scipy import stats
import random
import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)
# generez cate 10000 esantioane prin distrb exponentiale precizate
m1 = stats.expon(0, 1/4).rvs(10000) 
m2 = stats.expon(0, 1/6).rvs(10000)

p1 = 0.4 # cat serveste primul
p2 = 0.6 # cat serveste al doilea
x = p1 * m1 + p2 * m2 # timp total de servire client

mean = np.mean(x)
deviation = np.std(x)
print("Media timpului de servire:", mean)
print("Deviatia", deviation)

az.plot_posterior({'x':x})
plt.show()