import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# ex. 1

def posterior_grid(grid_points=50, heads=6, tails=9):
    grid = np.linspace(0, 1, grid_points)
    prior = abs(grid - 0.5) # prior = (grid <= 0.5).astype(int)
    likelihood = stats.binom.pmf(heads, heads+tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()
    return grid, posterior


def ex1(nr_heads=10, nr_tails=3):
    data = np.repeat([0, 1], (nr_tails, nr_heads))
    points = nr_heads + nr_tails
    h = data.sum()
    t = len(data) - h
    grid, posterior = posterior_grid(points, h, t)
    plt.plot(grid, posterior, 'o-')
    plt.title(f'heads = {h}, tails = {t}')
    plt.yticks([])
    plt.xlabel('θ - prob of heads')
    plt.show()

# ex. 2

def ex2(values_of_N):
    mean_err = []
    std_err = []
    for N in values_of_N:
        errors = []
        for _ in range(100):
            x, y = np.random.uniform(-1, 1, size=(2, N))
            inside = (x**2 + y**2) <= 1
            pi = inside.sum() * 4 / N
            error = abs((pi - np.pi) / pi) * 100
            errors.append(error)
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        mean_err.append(mean_error)
        std_err.append(std_error)

    plt.figure(figsize=(10, 6))
    plt.errorbar(values_of_N, mean_err, yerr=std_err, fmt='o-', capsize=5)
    plt.xscale('log')
    plt.xlabel('Nr of Points (N)', fontsize=12)
    plt.ylabel('Average Error (%)', fontsize=12)
    plt.title('Estimation of pi', fontsize=14)
    plt.legend(fontsize=10, loc='upper right')
    plt.show()

    # observam ca daca N creste, atat eroarea cat si deviatia scad; cu cat avem mai multe puncte, 
    # cu atat estimarea lui pi se apropie mai mult de valoarea reala


# ex. 3 
def metropolis(alpha, beta, draws=10000):
    trace = np.zeros(draws)
    old_x = 0.5
    old_prob = stats.beta.pdf(old_x, alpha, beta)
    delta = np.random.normal(0, 0.5, draws)

    for i in range(draws):
        new_x = old_x + delta[i]
        new_prob = stats.beta.pdf(new_x, alpha, beta)

        acceptance = new_prob / old_prob
        if acceptance >= np.random.random():
            trace[i] = new_x
            old_x = new_x
            old_prob = new_prob
        else:
            trace[i] = old_x

    return trace

def ex_3():
    alpha_prior, beta_prior = 2, 5 
    draws_metropolis = metropolis(alpha_prior, beta_prior)

    grid, posterior_grid_result = posterior_grid(heads=alpha_prior, tails=beta_prior)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(draws_metropolis[draws_metropolis > 0], bins=25, density=True, alpha=0.5, label='Metropolis')
    plt.xlabel('Theta')
    plt.ylabel('Density')
    plt.title('Metropolis - Beta(2, 5) Prior')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(grid, posterior_grid_result, label='Posterior Grid')
    plt.xlabel('Theta')
    plt.ylabel('Density')
    plt.title(f'Posterior Grid - heads={alpha_prior}, tails={beta_prior}')
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    #ex1()
    #ex1(55, 90)
    #ex2( [100, 1000, 10000])
    ex_3()

if __name__ == '__main__':
    main()