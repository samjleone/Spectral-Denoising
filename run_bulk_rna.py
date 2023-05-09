import scanpy as sc
import pandas as pd
from experiment_utils import generate_blob_data, add_noise_no_plot, get_graph, get_graph_magic, visuallize_graph, denoise_experiment, compare_denoised_signal
import numpy as np
from sklearn.datasets import make_blobs
## the output seems to be too much for this notebook, disabling tqdm.
from tqdm import tqdm
import magic
from functools import partialmethod

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True) 

## gene x cell. need to transpose
X = pd.read_csv('data/bulk_rna.csv', index_col='gene name').to_numpy()
# X = X - X.min() # make sure is positive
save_folder = 'results/bulk_rna/'
adata = sc.AnnData(X=X, dtype=X.dtype)
# Normalize and scale the data
# sc.pp.normalize_total(adata, target_sum=1e4)
# sc.pp.log1p(adata)
# sc.pp.scale(adata, max_value=10)

adata_gaussian = add_noise_no_plot(adata, noise_type='gaussian', mean=0, std=1, noise_scale=1)
adata_uniform = add_noise_no_plot(adata, noise_type='uniform', low=0.2, high=0.8)
adata_bernoulli = add_noise_no_plot(adata, noise_type='bernoulli', probability=0.5)

pd.DataFrame(adata_gaussian.X).to_csv(save_folder + 'noisy_gaussian.csv', index=False)
pd.DataFrame(adata_bernoulli.X).to_csv(save_folder + 'noisy_bernoulli.csv', index=False)
pd.DataFrame(adata_uniform.X).to_csv(save_folder + 'noisy_uniform.csv', index=False)

# pygsp_graph = get_graph(adata_gaussian)
pygsp_graph = get_graph_magic(adata_gaussian)
sig_denoised_gaussian = denoise_experiment(pygsp_graph, adata_gaussian.X, 'gaussian')
sig_denoised_gaussian_local_avg = denoise_experiment(pygsp_graph, adata_gaussian.X, 'local_average')
magic_op = magic.MAGIC()
sig_denoised_gaussian_magic = magic_op.fit_transform(adata_gaussian.X)
pd.DataFrame(sig_denoised_gaussian).to_csv(save_folder + 'denoised_gaussian.csv', index=False)
pd.DataFrame(sig_denoised_gaussian_local_avg).to_csv(save_folder + 'denoised_gaussian_local_avg.csv', index=False)
pd.DataFrame(sig_denoised_gaussian_magic).to_csv(save_folder + 'denoised_gaussian_magic.csv', index=False)

# pygsp_graph = get_graph(adata_bernoulli)
pygsp_graph = get_graph_magic(adata_bernoulli)
sig_denoised_bernoulli = denoise_experiment(pygsp_graph, adata_bernoulli.X, 'bernoulli')
sig_denoised_bernoulli_local_avg = denoise_experiment(pygsp_graph, adata_bernoulli.X, 'local_average')
magic_op = magic.MAGIC()
sig_denoised_bernoulli_magic = magic_op.fit_transform(adata_bernoulli.X)
pd.DataFrame(sig_denoised_bernoulli).to_csv(save_folder + 'denoised_bernoulli.csv', index=False)
pd.DataFrame(sig_denoised_bernoulli_local_avg).to_csv(save_folder + 'denoised_bernoulli_local_avg.csv', index=False)
pd.DataFrame(sig_denoised_bernoulli_magic).to_csv(save_folder + 'denoised_bernoulli_magic.csv', index=False)

pygsp_graph = get_graph_magic(adata_uniform)
sig_denoised_uniform = denoise_experiment(pygsp_graph, adata_uniform.X, 'uniform')
sig_denoised_uniform_local_avg = denoise_experiment(pygsp_graph, adata_uniform.X, 'local_average')
magic_op = magic.MAGIC()
sig_denoised_uniform_magic = magic_op.fit_transform(adata_uniform.X)
pd.DataFrame(sig_denoised_uniform).to_csv(save_folder + 'denoised_uniform.csv', index=False)
pd.DataFrame(sig_denoised_uniform_local_avg).to_csv(save_folder + 'denoised_uniform_local_avg.csv', index=False)
pd.DataFrame(sig_denoised_uniform_magic).to_csv(save_folder + 'denoised_uniform_magic.csv', index=False)

