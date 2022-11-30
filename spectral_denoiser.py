import scipy
import pygsp
import numpy as np
import torch
from torch import optim
from tqdm import tqdm

class Spectral_Denoiser:
    # Initialize
    # Params: Graph: a pygsp graph
    def __init__(self, Graph = None):
        if Graph != None:
            self.G = Graph
        else:
            raise ValueError("Graph Required")
    
    def remove_gaussian_noise(self, f_tilde, tau = 1):
        # Remove Gaussian Noise
        # Input: f_tilde, the observed signal
        # tau: eta * sigma^2 is a smoothing parameter
        # output: MLE original signal
        
        f_tilde = f_tilde.reshape(-1)
        gauss_filter = pygsp.filters.Filter(self.G, lambda x : 1/(1 + tau * x))
        f_check = gauss_filter.filter(f_tilde ,method = 'chebyshev')
        return f_check
        
    def remove_bernoulli_noise(self, f_tilde):
        # Remove Bernoulli Noise 
        # Solves the Dirichlet Problem on the Graph
        # Input: f_tilde: the observed signal
        
        # Output: f_check - the MLE estimate / Solution to the Heat Equation
        
        f_tild_vec = f_tilde.reshape(-1)
        
        # extrapolate where f_tilde is zero and nonzero
        I = np.where(f_tild_vec != 0)
        Ic = np.where(f_tild_vec == 0)
        sz_I = len(I[0])
        sz_Ic = len(Ic[0])
        
        # create relevant matrices
        Lap = self.G.L.todense()
        Adj = self.G.A.todense()*1
        subLap = Lap[Ic][:,Ic].reshape(sz_Ic,sz_Ic)
        subAdj = Adj[Ic][:,I].reshape(sz_Ic, sz_I)
        f_tild_I = f_tild_vec[I]
        
        # diffuse over the boundary and find the inverse
        delta = np.linalg.solve(subLap,(subAdj@f_tild_I).reshape(-1,1))
        delta = np.array(delta).reshape(-1)
        f_check = f_tild_vec.copy()
        
        # Impute corresponding Indices
        for i in range(sz_Ic):
            f_check[Ic[0][i]] = delta[i]
            
        return f_check
    
    def remove_uniform_noise(self, f_tilde, lr = 1, alpha = 1, beta = 1, gamma = 1, MAX_ITER = 500):
        
        #Removes Uniformly Distributed Noise
        # Inputs: 
        # 1. f_tilde - the observed signal
        # 2. alpha - a parameter controlling preference for smoothing
        # 3. beta - a parameter controlling for the size (log of abs. value) of estimates
        # 4. gamma - a parameter controlling preference for L2 distance of estimated factors to the uniform
        # 5. lr - learning rate for gradient descent
        
        signal = torch.tensor(f_tilde)
        L_tens = torch.tensor(self.G.L.todense())
        
        # normalize alpha, beta, and gamma to be invariant under rescaling the adjacencies
        # Heuristically Good
        alpha_ = 2/5 * alpha / (torch.norm(signal) * torch.trace(L_tens))
        beta_ = beta / torch.norm(signal) * 1e-2
        gamma_ = 5*1e-4*gamma
        
        # A function which takes a set of numbers between [0,1] f and approximates
        # its distance to the uniform distribution via the (squared) L2 norm between the Uniform Distribution and 
        # the Kernel Density Estimate of f (Approximates the Integral via Riemann Sums)
        
        def uniform_L2(f):

            K = 5000
            h = 2*1e-2
            n = f.shape[0]
            Riemann = torch.tensor(0, dtype = torch.double)

            def kde(z):
                return 1/n * torch.sum(1/torch.sqrt(torch.tensor(2*h**2*torch.pi)) * torch.exp(- (f - z)**2 / (2 * h**2)))

            grid = np.arange(-0.05,1.05,0.01)
            w = np.diff(grid)[0]

            for z in grid:
                u = 1
                if z < 0 or z > 1:
                    u = 0
                Riemann += w * (kde(z)-u)**2

            return Riemann
        
        # The Loss Function
        # Inputs:
        # 1. z - the proposed solution to the optimization
        # 2. signal - the observed signal
        
        def loss(z, signal):
            # Calculte the Laplacian Quadratic Form
            smoothness_prior = z.reshape(1,-1)@L_tens@z
            
            # The Posterior / Prior Probability of the Signal
            noise_prob = torch.sum(torch.log(torch.abs(z)))
            
            # f is the supposed set of ratios and will be distributed in [0,1]
            f = signal/z
            KL = uniform_L2(f)
            
            # add the respective (weighted) parts of the error
            return alpha_ * smoothness_prior + beta_ * noise_prob + gamma_ * KL
      
        # initialize our tensor 
        x = torch.tensor(signal.clone(), requires_grad = True)
        opt = optim.Adam([x], lr = lr)
        
        # Do the corresponding optimization
        for i in tqdm(range(MAX_ITER)):
            
            # calculate loss and perform the iterative step
            l = loss(x, signal)
            l.backward()
            opt.step()

            # Ensure Values are Strictly Between 0 and 1
            with torch.no_grad():
                x.clamp_(min=signal)
            
            # Zero the gradient
            opt.zero_grad()
            
            # If we have converged, stop
            if i > 1 and torch.norm(x_prev - x) < 1e-4:
                break
                
            # Keep Track of the Previous
            x_prev = x.clone().detach()
        
        # When Training is Over, Return
        return x.detach().numpy()