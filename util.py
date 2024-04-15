import numpy as np

def boltzmann_probability(utilities, temp):
    max_utility = np.maximum(*[u.astype(np.float32) for u in utilities])
    # Subtract the max utility to prevent overflow
    exp_diff = [np.exp((u - max_utility) / temp) for u in utilities]
    denom = sum(exp_diff)
    p_boltz = np.array([numerator / denom for numerator in exp_diff]).transpose()
    return p_boltz
