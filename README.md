# Spectral-Denoising

Code to perform spectral denoising of signals on Graphs, typically given a smoothness assumption on the signal. Optimizations include noise models such as:
* Additive Gaussian Noise
* Multiplicative Uniform Dropout
* Random Zeroing with Probability p

Code can be found in spectral_denoiser.py with examples in examples.ipynb. A typical call might look like: 

  denoiser = sd.Spectral_Denoiser(G)
  f_estimate = denoiser.remove_gaussian_noise(f_observed)
  f_estimate = denoiser.remove_bernoulli_noise(f_observed)
  f_estimate = denoiser.remove_uniform_noise(f_observed)
  
 Where G is a pygsp graph and f_observed is an array defined on the vertices of G.
