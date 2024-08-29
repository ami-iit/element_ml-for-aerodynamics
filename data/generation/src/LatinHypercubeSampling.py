import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

def generate_lhs_sampling(num_samples, ranges, seed):
    # Initialize the LHS sampler
    dimensions_number = len(ranges)
    sampler = qmc.LatinHypercube(d=dimensions_number, seed=seed)
    sample = sampler.random(n=num_samples)
    # Scale samples to the given ranges for the variables
    scaled_samples = np.zeros_like(sample)
    for i, range in enumerate(ranges):
        scaled_samples[:, i] = range[0] + sample[:, i] * (range[1] - range[0])
    return np.array(scaled_samples)
