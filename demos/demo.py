import numpy as np

from tsallis_stats.tsallis import q_gaussian

q, beta = 1.2, 3
mean, var, skew, kurt = q_gaussian.stats(q, beta, moments='mvsk')
print(mean, var)
print(q_gaussian.stats(q, beta, moments='m'))

rvs = q_gaussian.rvs(q, beta, size=10000)
print(rvs.mean(), rvs.var())