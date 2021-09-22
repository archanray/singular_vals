import numpy as np
import random
from tqdm import tqdm
from display_codes import display, display_precomputed_error, display_error_percentile
from get_dataset import get_data
from similarities import sigmoid, tps
import pickle
# import matplotlib.pyplot as plt

# labels and other hyperparameters
dataset_name = "random"
search_rank = 0

# load dataset
dataset, n = get_data(dataset_name)
d = dataset.shape[-1]

ratio = d / n

true_vals = np.linalg.svd(dataset, full_matrices=False, compute_uv=False)[search_rank]

print("starting approximation")

# perform approximation (assuming number of samples more than dimensions)
#"""
per_size_eps_mean = []
approximate_vals = []
approximate_vals_std = []
per_size_eps_lowp = []
per_size_eps_highp = []
for sample_size in tqdm(range(50, 1000, 10)):
    # approx_eigval_list = []
    per_round_eps = []
    approx_vals = []
    for trials in range(50):
        # print(trials)
        samples1 = np.sort(random.sample(range(n), sample_size))
        samples2 = np.sort(random.sample(range(d), int(sample_size*ratio)))
        sample_matrix = dataset[samples1][:,samples2]
        approx_val = (n*d) * np.linalg.svd(sample_matrix)[search_rank] / (sample_size * int(sample_size*ratio))
        approx_vals.append(approx_val)
        per_round_eps.append(np.abs(true_vals - approx_val) / sample_size*ratio)

    approximate_vals.append(np.mean(approx_vals))
    approximate_vals_std.append(np.std(approx_vals))
    per_size_eps_mean.append(np.mean(per_round_eps))
    # per_size_eps_max.append(np.max(per_round_eps))
    # per_size_eps_min.append(np.min(per_round_eps))
    # per_size_eps_std.append(np.std(per_round_eps))
    per_size_eps_lowp.append(np.percentile(per_round_eps, 20))
    per_size_eps_highp.append(np.percentile(per_round_eps, 80))
#"""

display(dataset_name, [true_vals], n, search_rank, \
            approximate_vals, approximate_vals_std, 1000, 50)

display_precomputed_error(dataset_name, per_size_eps_mean, d, \
                              search_rank, 1000, error_std=[], \
                              percentile1=per_size_eps_lowp, \
                              percentile2=per_size_eps_highp, log=True, min_samples=50)
