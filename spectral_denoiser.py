import scipy
import pygsp
import numpy as np
import torch
from torch import optim
from tqdm import tqdm

class Spectral_Denoiser:
    def __init__(self, Graph = None):
        if Graph != None:
            self.G = Graph
        else:
            raise ValueError("Graph Required")
            
    def gaussian_noise(self, f_tilde, tau = 1):
        f_tilde = f_tilde.reshape(-1)
        gauss_filter = pygsp.filters.Filter(self.G, lambda x : 1/(1 + tau * x))
        f_check = gauss_filter.filter(f_tilde ,method = 'chebyshev')
        return f_check
        
    def bernoulli_noise(self, f_tilde):
        f_tild_vec = f_tilde.reshape(-1)
        I = np.where(f_tild_vec != 0)
        Ic = np.where(f_tild_vec == 0)
        sz_I = len(I[0])
        sz_Ic = len(Ic[0])
        Lap = self.G.L.todense()
        Adj = self.G.A.todense()*1
        subLap = Lap[Ic][:,Ic].reshape(sz_Ic,sz_Ic)
        subAdj = Adj[Ic][:,I].reshape(sz_Ic, sz_I)
        f_tild_I = f_tild_vec[I]
        delta = np.linalg.solve(subLap,(subAdj@f_tild_I).reshape(-1,1))
        delta = np.array(delta).reshape(-1)
        f_check = f_tild_vec.copy()
        
        for i in range(sz_Ic):
            f_check[Ic[0][i]] = delta[i]
            
        return f_check
    
    def uniform_noise(self, f_tilde, lr = 1, alpha = 1, beta = 1, gamma = 1, MAX_ITER = 500):
        
        signal = torch.tensor(f_tilde)
        L_tens = torch.tensor(self.G.L.todense())
        alpha_ = 2/5 * alpha / (torch.norm(signal) * torch.trace(L_tens))
        beta_ = beta / torch.norm(signal) * 1e-2
        gamma_ = 5*1e-4*gamma
        
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
        
        def loss(z, signal):
            smoothness_prior = z.reshape(1,-1)@L_tens@z
            noise_prob = torch.sum(torch.log(torch.abs(z)))

            f = signal/z
            KL = uniform_L2(f)
            #print(alpha_ * smoothness_prior, beta_ * noise_prob, gamma_ * KL)
            return alpha_ * smoothness_prior + beta_ * noise_prob + gamma_ * KL
      
        x = torch.tensor(signal.clone().detach(), requires_grad = True)
        opt = optim.Adam([x], lr = lr)
        
        for i in tqdm(range(MAX_ITER)):
            l = loss(x, signal)
            l.backward()
            opt.step()

            with torch.no_grad():
                x.clamp_(min=signal)
                
            opt.zero_grad()
            
            if i > 1 and torch.norm(x_prev - x) < 1e-4:
                break
            x_prev = x.clone().detach()
                
        return x