import numpy as np
from random import *
from scipy import stats
import arviz as az
import matplotlib.pyplot as plt

np.random.seed(1)

ss = []
sb = []
bs = []
bb = []

for i in range(100): # simulez 100 experimente
    ss_c = 0 # stema stema contor
    sb_c = 0 # stema ban contor
    bs_c = 0
    bb_c = 0
    for j in range(10): # 10 aruncari de monezi
        r1 = randint(1, 100) # r1 < 50 => ban, altfel stema
        r2 = randint(1, 100)
        if r1 < 50 and r2 < 30 :
            ss_c += 1
        elif r1 < 50 and r2 >= 30:
            sb_c += 1
        elif r1 >= 50 and r2 < 30:
            bs_c += 1
        else:
            bb_c += 1
    ss.append(ss_c)
    sb.append(sb_c)
    bs.append(bs_c)
    bb.append(bb_c)

az.plot_posterior({'ss':ss, 'bs':bs, 'sb':sb,'bb':bb})
plt.show()

