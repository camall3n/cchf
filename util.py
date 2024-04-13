import numpy as np

def boltzmann_probability(utility1, utility2, temp):
    max_utility = np.maximum(utility1, utility2)
    # Subtract the max utility to prevent overflow
    exp_diff1 = np.exp((utility1 - max_utility) / temp)
    exp_diff2 = np.exp((utility2 - max_utility) / temp)
    return exp_diff1 / (exp_diff1 + exp_diff2)
