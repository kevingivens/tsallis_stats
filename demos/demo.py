import matplotlib.pyplot as plt
import numpy as np

from tsallis_stats.tsallis import q_gaussian

if __name__ == "__main__":
    q, beta = 1.2, 3
    mean, var, skew, kurt = q_gaussian.stats(q, beta, moments='mvsk')
    
    print(mean, var)
    # print(q_gaussian.stats(q, beta, moments='m'))
    #fig, ax = plt.subplots(1, 1)
    #x = np.linspace(q_gaussian.ppf(0.01, q, beta), q_gaussian.ppf(0.99, q, beta), 100)
    #ax.plot(x, q_gaussian.pdf(x, q, beta, 'r-', lw=5, alpha=0.6, label='q_gaussian pdf'))
    rvs = q_gaussian.rvs(q, beta, size=10000)
    print(rvs.mean(), rvs.var())
    #ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
    #ax.legend(loc='best', frameon=False)
    #plt.show()