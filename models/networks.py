# Reference upstream repository: https://github.com/longinyu/ksivi
# This file is part of the KPG overlay package.

import torch
import torch.nn as nn


class SIMINet(nn.Module):
    def __init__(self, namedict, device):
        super(SIMINet, self).__init__()
        self.z_dim = namedict.z_dim
        self.h_dim = namedict.h_dim
        self.out_dim = namedict.out_dim

        self.mu = nn.Sequential(
            nn.Linear(self.z_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.out_dim)
        )
        self.log_var = nn.Parameter(torch.zeros(namedict.out_dim) + namedict.log_var_ini, requires_grad = True)
        self.device = device
        self.log_var_min = namedict.log_var_min
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(mu)
        return mu + std * eps, eps/std
    
    def getmu(self, Z):
        return self.mu(Z)
    
    def getstd(self):
        log_var = self.log_var.clamp(min = self.log_var_min)
        std = torch.exp(log_var/2)
        return std
    
    def forward(self, Z):
        mu = self.mu(Z)
        log_var = self.log_var.clamp(min = self.log_var_min)
        X, neg_score_implicit = self.reparameterize(mu, log_var)
        return X, neg_score_implicit

    def sampling(self, num = 1000, sigma = 1):
        with torch.no_grad():
            Z = torch.randn([num, self.z_dim], ).to(self.device)
            Z = Z * sigma
            X, _ = self.forward(Z)
        return X
    
    
# [KPG block] Proposal distribution module introduced for path-gradient variants.
class NormalProposalDistribution(torch.nn.Module):
    def __init__(self, inp_dim, out_dim, hidden_dim, num_layers, device, alpha_inf = 0.01, alpha_sup = 0.9):# alpha_inf = 0.01, alpha_sup = 0.9):
        super().__init__()
        layers = []
        
        layers.append(nn.Linear(inp_dim, hidden_dim))
        layers.append(nn.ReLU())  
        
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        self.base_model = nn.Sequential(*layers).to(device)

        self.mean_model  = nn.Sequential(nn.Linear(hidden_dim, out_dim)).to(device)
        self.std_model   = nn.Sequential(nn.Linear(hidden_dim, out_dim),
                                        nn.Softplus()).to(device)
        self.alpha_model = nn.Sequential(nn.Linear(hidden_dim, 1),
                                        nn.Sigmoid()).to(device)
        
        self.out_dim = out_dim
        self.alpha_inf = alpha_inf
        self.alpha_sup = alpha_sup

        self.std_normal = torch.distributions.Normal(torch.zeros([1, 1, 1], device=device),
                                                     torch.ones([1, 1, 1], device=device))
    
    def forward(self, z):
        base = self.base_model(z)
        return self.mean_model(base), self.std_model(base), self.alpha_inf  + (self.alpha_sup-self.alpha_inf)*self.alpha_model(base)

    def sample(self, z, k):
        n = z.shape[0]
        
        mean, std, alpha = self(z)

        dist = torch.distributions.Normal(mean, std)
        samples = dist.rsample([k])
        
        idx = torch.rand(k, n, device=z.device)  < alpha.transpose(0, 1).detach()
        
        samples_shared = torch.randn([k, self.out_dim], device=z.device)
        samples[idx] = samples_shared[:, None, :].expand([k, n, self.out_dim])[idx]
        
        #samples[idx] = torch.randn([idx.sum(), self.out_dim], device=z.device)         

        return samples, self.inner_log_prob(self.std_normal.log_prob(samples), dist.log_prob(samples), alpha), samples_shared, idx
    
    def inner_log_prob(self, log_peps, log_pepsz, alpha):
        #print(alpha.mean())
        return torch.logaddexp(alpha.log().transpose(0, 1) + log_peps.sum(-1),
                                (1-alpha).log().transpose(0, 1) + log_pepsz.sum(-1))

    def log_prob(self, eps, z):
        mean, std, alpha = self(z)

        dist = torch.distributions.Normal(mean[None, ...], std[None, ...])
        log_pepsz = dist.log_prob(eps)

        log_peps = self.std_normal.log_prob(eps)

        #print(self.inner_log_prob(log_peps, log_pepsz, alpha))
        #print(self.inner_log_prob(log_peps, log_pepsz, alpha).isnan().any())
        #print((alpha == self.alpha_sup).any())
        
        return self.inner_log_prob(log_peps, log_pepsz, alpha)
