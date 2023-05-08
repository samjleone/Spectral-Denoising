import scanpy as sc
import numpy as np
from sklearn.datasets import make_blobs
import pygsp
import matplotlib.pyplot as plt
import spectral_denoiser
import magic
import phate

def generate_blob_data(n_cells=1000, n_genes=500, n_clusters=5, cluster_std=8.0, center_box=(-5, 5)):
    # Generate single-cell RNA-seq data using blobs
    data, _ = make_blobs(n_samples=n_cells, n_features=n_genes, centers=n_clusters, cluster_std=cluster_std, center_box=center_box)
    data -= data.min()
    data = data.astype(np.float32)
    # data = data.astype(np.float64)

    # Create an AnnData object with the generated data
    adata = sc.AnnData(X=data, dtype=data.dtype)

    # Normalize and scale the data
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)

    return adata

def add_noise(adata, noise_type='gaussian', **kwargs):
    """
    Plots the input adata and adata with noise, along with their histograms and scatter plot.
    
    Parameters:
    adata (anndata.AnnData): Input adata object.
    noise_type (str): Type of noise to add. 'gaussian' (default), 'uniform', or 'bernoulli'.
    **kwargs: Additional arguments for the chosen noise type.
        - For Gaussian noise: mean (float), std (float), noise_scale (float)
        - For uniform noise: low (float), high (float)
        - For Bernoulli noise: probability (float)
    
    Returns:
    adata_with_noise (anndata.AnnData): adata with added noise
    """
    # Make a copy of the input data
    adata_with_noise = adata.copy()
    
    # Add noise based on the chosen noise type
    if noise_type == 'gaussian':
        mean = kwargs.get('mean', 0)
        std = kwargs.get('std', 1)
        noise_scale = kwargs.get('noise_scale', 1)
        gaussian_noise = np.random.normal(loc=mean, scale=std*noise_scale, size=adata.X.shape)
        adata_with_noise.X += gaussian_noise
    elif noise_type == 'uniform':
        low = kwargs.get('low', 0)
        high = kwargs.get('high', 1)
        uniform_noise = np.random.uniform(low=low, high=high, size=adata.X.shape)
        adata_with_noise.X *= uniform_noise
    elif noise_type == 'bernoulli':
        probability = kwargs.get('probability', 0.5)
        bernoulli_noise = np.random.binomial(n=1, p=probability, size=adata.X.shape)
        adata_with_noise.X *= bernoulli_noise
    else:
        raise ValueError(f"Invalid noise type: {noise_type}")
    
    # Create a figure with 2 rows and 2 columns of subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Plot the first histogram on the first subplot
    axs[0, 0].hist(adata.X.flatten(), 30)
    axs[0, 0].set_title("Histogram of adata.X")

    # Plot the second histogram on the second subplot
    axs[0, 1].hist(adata_with_noise.X.flatten(), 30)
    axs[0, 1].set_title(f"Histogram of adata_with_{noise_type}_noise.X")

    # Plot the third histogram on the third subplot
    if noise_type == 'gaussian':
        axs[1, 0].hist(gaussian_noise.flatten(), 30)
    elif noise_type == 'uniform':
        axs[1, 0].hist(uniform_noise.flatten(), 30)
    elif noise_type == 'bernoulli':
        axs[1, 0].hist(bernoulli_noise.flatten(), 30)
    axs[1, 0].set_title(f"Histogram of {noise_type}_noise")

    # Add a scatter plot on the last subplot
    axs[1, 1].scatter(adata.X.flatten(), adata_with_noise.X.flatten(), s=1, alpha=0.1)
    axs[1, 1].set_xlabel("adata.X")
    axs[1, 1].set_ylabel(f"adata_with_{noise_type}_noise.X")
    axs[1, 1].set_title(f"Scatter plot of adata.X vs adata_with_{noise_type}_noise.X")

    # Add a centered suptitle
    plt.suptitle(f"Plots of Data with and without {noise_type.capitalize()} Noise")

    # Adjust the spacing between subplots to prevent overlap
    plt.tight_layout()

    # Display the plot
    plt.show()

    return adata_with_noise

def get_graph(adata, k=10):
    # Perform PCA to reduce the dimensionality
    sc.tl.pca(adata)

    # Compute k-nearest neighbors and Gaussian kernel weights
    # k = 10  # You can change this value to the desired number of neighbors
    sc.pp.neighbors(adata, n_neighbors=k, use_rep='X_pca', method='gauss', metric='euclidean', knn=True)

    # The resulting k-NN graph is stored in the AnnData object's 'connectivities' attribute
    graph = adata.uns['neighbors']['connectivities']

    coordinates = adata.obsm['X_pca'][:, :2]
    pygsp_graph = pygsp.graphs.Graph(graph)
    pygsp_graph.set_coordinates(coordinates)
    return pygsp_graph

def get_graph_magic(adata):
    magic_op = magic.MAGIC()
    magic_op.fit(adata.X)
    pygsp_graph = magic_op.graph.to_pygsp()
    phate_op = phate.PHATE()
    data_phate = phate_op.fit_transform(adata)
    coordinates = data_phate
    pygsp_graph.set_coordinates(coordinates)
    return pygsp_graph

def visuallize_graph(pygsp_graph):
    ## visuallize
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the graph
    pygsp.plotting.plot_graph(pygsp_graph, ax=ax, vertex_size=5)

    # Set axis properties
    ax.set_title('Gaussian Kernel k-NN Graph of scRNA-seq data')
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.axis('equal')

    # Show the plot
    plt.show()

def denoise_experiment(pygsp_graph, signal_noisy, noise_type):
    """_summary_

    Args:
        pygsp_graph (_type_): _description_
        signal_noisy (_type_): _description_
        noise_type (_type_): shape (n, m) or (n, ). N is number of nodes, m is number of channels (signals)

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if len(signal_noisy.shape) == 1:
        signal_noisy = signal_noisy.reshape(-1, 1)
    Denoising_Machine = spectral_denoiser.Spectral_Denoiser(pygsp_graph)
    if noise_type == 'gaussian':
        signal_denoised = Denoising_Machine.remove_gaussian_noise(signal_noisy)
    elif noise_type == 'bernoulli':
        signal_denoised = Denoising_Machine.remove_bernoulli_noise(signal_noisy, method = 'approximate', time = 500)
    elif noise_type == 'uniform':
        signal_denoised = Denoising_Machine.remove_uniform_noise(signal_noisy, method='pg')
    else: raise ValueError('noise type not recognized')
    return signal_denoised

def compare_denoised_signal(sig_true, sig_noisy, sig_denoised):
    # Create a figure with 1 row and 3 columns of subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Calculate the MSE between sig_true and sig_true_with_gaussian_noise
    mse1 = np.mean((sig_true - sig_noisy)**2)
    # Add a scatter plot of sig_true vs. sig_noisy on the first subplot
    axs[0].scatter(sig_true.flatten(), sig_noisy.flatten(), s=1, alpha=0.1)
    axs[0].set_xlabel("sig_true")
    axs[0].set_ylabel("sig_noisy")
    axs[0].set_title(f"Scatter plot with MSE={mse1:.2f}")

    # Calculate the MSE between signal (denoised) and sig_true
    mse2 = np.mean((sig_denoised - sig_true)**2)
    # Add a scatter plot of sig_denoised vs. sig_true on the second subplot
    axs[1].scatter(sig_true.flatten(), sig_denoised.flatten(), s=1, alpha=0.1)
    axs[1].set_xlabel("sig_true")
    axs[1].set_ylabel("sig_denoised")
    axs[1].set_title(f"Scatter plot with MSE={mse2:.2f}")

    # Calculate the MSE between signal (denoised) and sig_noisy
    mse3 = np.mean((sig_denoised - sig_noisy)**2)
    # Add a scatter plot of signal (denoised) vs. sig_noisy on the third subplot
    axs[2].scatter(sig_denoised.flatten(), sig_noisy.flatten(), s=1, alpha=0.1)
    axs[2].set_xlabel("Signal (denoised)")
    axs[2].set_ylabel("sig_noisy")
    axs[2].set_title(f"Scatter plot with MSE={mse3:.2f}")

    # Add a centered suptitle
    plt.suptitle("Scatter Plots of True Signal, Noisy Signal, and Denoised Signal")

    # Adjust the spacing between subplots to prevent overlap
    plt.tight_layout()

    # Display the plot
    plt.show()

    return {'MSE noisy':mse1, 'MSE denoised':mse2, 'MSE denoised vs noisy':mse3}
