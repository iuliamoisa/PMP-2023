import matplotlib.pyplot as plt
import numpy as np 
from scipy import stats

def posterior_grid(grid_points=50, success=5):
    """
    A grid implementation for the coin-flipping problem
    """
    grid = np.linspace(0, 1, grid_points)
    prior = np.repeat(1/grid_points, grid_points) # uniform prior
    likelihood = stats.geom.pmf(success, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior

points = 100
success = 5
grid, posterior = posterior_grid(points, success)
plt.plot(grid, posterior, 'o-')
plt.title(f'Succes = {success}')
plt.yticks([])
plt.xlabel('θ')
plt.show()

# Pt a modela prima aparitie a unei steme, folosim distributia geometrica 
# in loc de cea  binomiala  in calculul verosimilitatii.
# Distributia geometrica modeleaza nr de incercari necesare pt a obtine primul "succes"
# intr-o serie de incercari bernoulli
# In cazul dat, "succes"-ul il reprezinta obtinerea unei steme la o aruncare.