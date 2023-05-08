import scipy
import pygsp
import numpy as np
# import torch
# from torch import optim
from tqdm import tqdm
import warnings
import cvxopt as opt
from cvxopt import matrix, spmatrix, sparse
from cvxopt.solvers import qp, options
from cvxopt import blas

class Spectral_Denoiser:
    # Initialize
    # Params: Graph: a pygsp graph
    def __init__(self, Graph = None):
        if Graph != None:
            self.G = Graph
            self.P = None
        else:
            raise ValueError("Graph Required")
            
    def bandlimit_low(self, f_tilde, omega=-1, k=-1):
        # Remove Gaussian Noise
        # Input: f_tilde, the observed signal
        # tau: eta * sigma^2 is a smoothing parameter
        # output: MLE original signal

        if omega > -1:
            f_tilde = f_tilde.reshape(-1)
            gauss_filter = pygsp.filters.Filter(self.G, lambda x : 1*(x <= omega))
            f_check = gauss_filter.filter(f_tilde ,method = 'chebyshev')
            return f_check
        elif k > -1:
            G = self.G
            G.compute_fourier_basis()
            self.Psi = G.U
            self.evals = G.e
            omega = self.evals[k]
            gauss_filter = pygsp.filters.Filter(self.G, lambda x : 1*(x <= omega))
            f_check = gauss_filter.filter(f_tilde ,method = 'exact')
            return f_check
        else:
            raise ValueError("Can Only Filter Based on Bandwidth")
            
        f_tilde = f_tilde.reshape(-1)
        gauss_filter = pygsp.filters.Filter(self.G, lambda x : 1/(1 + tau * x))
        f_check = gauss_filter.filter(f_tilde ,method = 'chebyshev')
        return f_check
    
    def bandlimit_high(self, f_tilde, omega=-1, k=-1):
        # Remove Gaussian Noise
        # Input: f_tilde, the observed signal
        # tau: eta * sigma^2 is a smoothing parameter
        # output: MLE original signal

        if omega > -1:
            f_tilde = f_tilde.reshape(-1)
            gauss_filter = pygsp.filters.Filter(self.G, lambda x : 1*(x >= omega))
            f_check = gauss_filter.filter(f_tilde ,method = 'chebyshev')
            return f_check
        elif k > -1:
            G = self.G
            G.compute_fourier_basis()
            self.Psi = G.U
            self.evals = G.e
            omega = self.evals[-k]
            gauss_filter = pygsp.filters.Filter(self.G, lambda x : 1*(x >= omega))
            f_check = gauss_filter.filter(f_tilde ,method = 'exact')
            return f_check
        else:
            raise ValueError("Can Only Filter Based on Bandwidth")
            
        f_tilde = f_tilde.reshape(-1)
        gauss_filter = pygsp.filters.Filter(self.G, lambda x : 1/(1 + tau * x))
        f_check = gauss_filter.filter(f_tilde ,method = 'chebyshev')
        return f_check
    
    def local_average(self, f_tilde, t=1):
        
        if self.P is None:
            G = self.G
            A = G.W.todense()
            P = np.diag(1/np.array(np.sum(A,axis=0))[0])@A
            self.P = P
            
        for t_ in range(t):
            f_tilde = self.P@f_tilde
        
        return np.array(f_tilde)
    
    def nuclear_norm_appx(self, f_tilde, tau):
        restored = np.zeros(f_tilde.shape)
        for i in range(f_tilde.shape[2]):
            M = f_tilde[:,:,i]
            nrow = M.shape[0]
            ncol = M.shape[1]
            U, S, VT = np.linalg.svd(M)
            S_ = [np.sign(s)*max(0, np.abs(s) - tau) for s in S]
            restored[:,:,i] = U@np.diag(S_)@VT
        return restored
    
    def remove_gaussian_noise(self, f_tilde, tau = None):
        # Remove Gaussian Noise
        # Input: f_tilde, the observed signal
        # tau: eta * sigma^2 is a smoothing parameter
        # output: MLE original signal
        
        if tau == None:
            tau = self.estimate_tau_for_gaussian(f_tilde)
        
        if tau < 0:
            tau = 1
        
        gauss_filter = pygsp.filters.Filter(self.G, lambda x : 1/(1 + tau * x))
        f_check = gauss_filter.filter(f_tilde ,method = 'chebyshev')
        return f_check
        
    def remove_bernoulli_noise(self, f_tilde, method = 'exact', time = 250):
        # Remove Bernoulli Noise 
        # Solves the Dirichlet Problem on the Graph
        # Input: f_tilde: the observed signal
        
        # Output: f_check - the MLE estimate / Solution to the Heat Equation
        
        restored = np.zeros(f_tilde.shape)
        for signal_index in range(f_tilde.shape[1]):
            f_tild_vec = f_tilde[:,signal_index].reshape(-1)
            n = self.G.N

            # extrapolate where f_tilde is zero and nonzero
            I = np.where(f_tild_vec != 0)
            Ic = np.where(f_tild_vec == 0)
            sz_I = len(I[0])
            sz_Ic = len(Ic[0])

            # get Adjacency matrix

            Adj = self.G.A.todense()*1

            if method == 'exact':
                if sz_I > 10000:
                    warnings.warn("Approximation is recommended for large problems")

                # create relevant matrices
                Lap = self.G.L.todense()

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

            elif method == 'approximate':
                x = f_tild_vec.copy()
                # initialize walk matrix
                P = np.zeros((n,n))
                for i in range(n):
                    if f_tild_vec[i] != 0:
                        # Stabilize at the known vertices
                        P[i,i] = 1
                    else:
                        # Diffuse Otherwise
                        P[i,:] = Adj[i,:]/np.sum(Adj[i,:])

                # iterate
                for iter in tqdm(range(time)):
                    x_prev = x.copy()
                    x = P@x
                    if np.linalg.norm(x_prev-x) < 1e-4:
                        break

                f_check = x
            else:
                raise Exception("Methods are 'approximate' or 'exact'")
            
            restored[:,signal_index] = f_check.reshape(-1)
        
        return restored
    
    def estimate_tau_for_gaussian(self, tilde_samples):
        Lap = self.G.L.todense()
        n = Lap.shape[0]
        num_samps = tilde_samples.shape[1]
        
        order_1 = 0
        order_2 = 0
        
        for i in range(num_samps):
            theta_tilde = tilde_samples[:,i]
            order_1 += theta_tilde.T@Lap@theta_tilde
            order_2 += np.linalg.norm(Lap@theta_tilde)**2
        
        order_1 = order_1/num_samps; order_2 = order_2/num_samps

        numerator = (n-1)*order_2 - np.trace(Lap)*order_1
        denominator = (np.linalg.norm(Lap)**2)*order_1 - np.trace(Lap)*order_2
        prop_tau = numerator/denominator
        
        if prop_tau > 0:
            return prop_tau
        else:
            return 1
    
    def estimate_eta_for_uniform(self, tilde_samples, Linv = None):
        Lap = self.G.L.todense()
        n = Lap.shape[0]
        num_samps = tilde_samples.shape[1]
        
        if Linv is None:
            Linv = np.linalg.pinv(Lap)

        avg_t = 0
        for i in range(num_samps):
            avg_t += np.linalg.norm(tilde_samples[:,i])**2

        avg_t = avg_t/num_samps

        mu_sq = (2*np.sqrt(n)*np.mean(tilde_samples))**2
        # print(mu_sq)

        numerator = np.trace(Linv)
        denominator = 6*avg_t - 2*mu_sq

        return numerator/denominator
    
    def remove_uniform_noise(self, theta_tilde, max_iter = 10, true_signal = None, max_iter_qp=20, epsilon = 1e-3, bump=1, eta=None, method = 'ccp', lr = 1, verbose=False):
        
        Lap = self.G.L.todense()
        n = Lap.shape[0]

        # perform method of moments estimation for eta

        if eta == None:
            eta = self.estimate_eta_for_uniform(theta_tilde)
            if verbose == True:
                print("Value of eta: " + str(eta))
        
        def convex_subroutine(L_sub, theta_tilde_sub, x_k):
            options['show_progress'] = False
            options["maxiters"] = max_iter_qp

            theta_tilde_sub = theta_tilde_sub.reshape(-1)

            n = max(theta_tilde_sub.shape)

            Uniform_Enforcer_Matrix = np.zeros((n,n))
            Uniform_Enforcer_Vector = -np.ones((n,1)).reshape(-1,1)

            for i in range(n):
                Uniform_Enforcer_Matrix[i,i] = - 1/theta_tilde_sub[i]

            concave_gradient = (1/np.abs(x_k)).reshape(-1,1)
            x_k_plus_1 = qp(matrix(L_sub.astype(np.double)), matrix(concave_gradient), matrix(Uniform_Enforcer_Matrix), matrix(Uniform_Enforcer_Vector))['x']
            
            return np.array(x_k_plus_1).reshape(-1,1)
        
        def primal_cost(x):
            x = x.reshape(-1,1)
            return eta*x.T@Lap@x + np.sum(np.log(np.abs(x)))
        
        if true_signal is None:
            print("Truth unknown")
        else:
            true_cost = primal_cost(true_signal)
            print(f'True cost: {true_cost}')
        
        # Optimize for every signal 

        restored_signals = np.zeros((theta_tilde.shape[0],theta_tilde.shape[1]))
        # print(theta_tilde.shape[1])
        for signal_index in np.arange(theta_tilde.shape[1]):
            
            noisy_signal = theta_tilde[:,signal_index].reshape(-1,1)

            x0 = 1.05*noisy_signal

            prev_cost = primal_cost(x0)
            relative_stopping_criteria = primal_cost(x0)*epsilon
            for i in range(n):
                x0[i] = x0[i] + bump*np.sign(x0[i])
            losses = []
            x_k = x0
            
            i = 0
            delta = relative_stopping_criteria + 1
            new_cost = prev_cost + relative_stopping_criteria + 1
            
            while (i < max_iter) and np.abs(delta) > relative_stopping_criteria:
          
                prev_cost = new_cost
                losses.append(primal_cost(x_k))
                
                if method == 'ccp':
                    x_k = convex_subroutine(eta*Lap, noisy_signal, x_k)
                    
                elif method == 'pg':
                    x_k = x_k.reshape(-1,1)
                    grad = 2*eta*Lap@x_k + 1/x_k
                    x_k = x_k - lr*grad
                    for j in range(n):
                        if noisy_signal[j] == 0:
                            x_k[j] = 0
                        elif x_k[j]/noisy_signal[j] < 1:
                            x_k[j] = noisy_signal[j]
                    
                else:
                    raise ValueError("Method should be ccp (Convex-Concave Procedure) or pg (Projected Gradient)")
                new_cost = primal_cost(x_k)
                delta = prev_cost - new_cost
                i += 1
                
            if verbose == True:
                print(f"Final Cost: {new_cost} Achieved in {i} Iterations")
            restored_signals[:,signal_index] = x_k.reshape(-1)
            
        return restored_signals
    
    def remove_uniform_noise_old(self, f_tilde, lr = 1, alpha = 1, beta = 1, gamma = 1, MAX_ITER = 500):
        
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
            if gamma > 0:
                KL = uniform_L2(f)
            else:
                KL = 0
            
            # add the respective (weighted) parts of the error
            return alpha_ * smoothness_prior + beta_ * noise_prob + gamma_ * KL
      
        # initialize our tensor 
        x = torch.tensor(signal.clone().detach(), requires_grad = True)
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