# Reference upstream repository: https://github.com/longinyu/ksivi
# This file is part of the KPG overlay package.
# Derived from KSIVI experiment scripts (primarily sivistein_* variants).
# Changes are minimal and focused on integrating the KPG method/components.


import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch

from models.networks import *
from models.target_models import target_distribution
from tqdm import tqdm
from utils.annealing import annealing
from utils.parse_config import parse_config
import logging
from utils.kernels import *

import time

class SIVIPathsteinISLangevin(object):
    def __init__(self, config):
        self.config = config
        self.device = self.config.device
        self.target = self.config.target_score
        self.trainpara = self.config.train
        self.num_iters = self.trainpara.num_perepoch * self.config.train.num_epochs
        self.iter_idx = 0
        self.kernel = {"gaussian":gaussian_kernel, "laplace":laplace_kernel,"IMQ":IMQ_kernel}[self.config.kernel]
        
    def preprocess(self):
        self.save_samples_to_path = os.path.join(self.config.log_path, "traceplot")
        os.makedirs(self.save_samples_to_path,exist_ok=True)
    
    def loaddata(self):
        self.target_model = target_distribution[self.target](num_interval = self.config.num_interval, num_obs = self.config.num_obs, beta = self.config.beta, T = self.config.T, sigma = self.config.sigma, device = self.device)
    def learn(self):
        self.preprocess()
        self.loaddata()

        sgld_samples = torch.load("sgld_lagevin.pt").to(self.device)
        for k in range(3):
            prop = NormalProposalDistribution(self.trainpara.out_dim, self.trainpara.z_dim, self.trainpara.h_dim, 2, self.device)
            std_normal = torch.distributions.Normal(torch.zeros([1, 1], device=self.device), torch.ones([1, 1], device=self.device))


            self.SemiVInet = SIMINet(self.trainpara, self.device).to(self.device)
            annealing_coef = lambda t: annealing(t, warm_up_interval = self.num_iters//self.trainpara.warm_ratio, anneal = self.trainpara.annealing)

            optimizer_VI = torch.optim.Adam([{'params':self.SemiVInet.mu.parameters(),'lr': self.trainpara.lr_SIMI},
                                  {'params':self.SemiVInet.log_var,'lr': self.trainpara.lr_SIMI_var}], betas=(.9, .99))
            optimizer_prop = torch.optim.Adam(prop.parameters(), lr = self.trainpara.lr_SIMI_var)
            scheduler_VI = torch.optim.lr_scheduler.StepLR(optimizer_VI, step_size=self.trainpara.gamma_step, gamma=self.trainpara.gamma)

            loss_list = []
            mll = []
            for epoch in tqdm(range(1, self.trainpara.num_epochs+1)):
                if (epoch-1) ==0:
                    X = self.SemiVInet.sampling(num=self.config.sampling.num)
                    figname = f'{self.iter_idx+1}.jpg'
                    self.target_model.trace_plot(X, figpath=self.save_samples_to_path, figname=figname)
                for i in range(1, self.trainpara.num_perepoch+1):
                    self.iter_idx = (epoch-1) * self.trainpara.num_perepoch + i

                    #start = time.time()

                    Z = torch.randn([self.trainpara.batchsize, self.trainpara.z_dim]).to(self.device)
                    Z_aux = torch.randn([self.trainpara.batchsize, self.trainpara.z_dim]).to(self.device)

                    X, neg_score_implicit = self.SemiVInet(Z)

                    optimizer_prop.zero_grad()
                    prop_loss = -prop.log_prob(Z.detach()[None, ...], X.detach()).mean()
                    prop_loss.backward()
                    optimizer_prop.step()

                    with torch.no_grad():

                        Z_aux, lps, Z_shared, idx = prop.sample(X.detach(), self.trainpara.batchsize)
                        nidx = idx.logical_not()

                        X_aux_shared, neg_score_implicit_aux_shared = self.SemiVInet(Z_shared)
                        compu_targetscore_aux_shared = self.target_model.score(X_aux_shared) * annealing_coef(self.iter_idx)


                        neg_score_implicit_aux_shared = neg_score_implicit_aux_shared[:, None, :].expand(-1, lps.shape[1], -1)
                        compu_targetscore_aux_shared = compu_targetscore_aux_shared[:, None, :].expand(-1, lps.shape[1], -1)
                        X_aux_shared = X_aux_shared[:, None, :].expand(-1, lps.shape[1], -1)

                        X_aux, neg_score_implicit_aux = self.SemiVInet(Z_aux[nidx])
                        compu_targetscore_aux = self.target_model.score(X_aux) * annealing_coef(self.iter_idx)

                        neg_score_implicit_aux_all = torch.empty([Z_aux.shape[0], Z_aux.shape[1], X_aux_shared.shape[2]], device = Z_aux.device)
                        neg_score_implicit_aux_all[idx] = neg_score_implicit_aux_shared[idx]
                        neg_score_implicit_aux_all[nidx] = neg_score_implicit_aux        

                        compu_targetscore_aux_all = torch.empty([Z_aux.shape[0], Z_aux.shape[1], X_aux_shared.shape[2]], device = Z_aux.device)
                        compu_targetscore_aux_all[idx] = compu_targetscore_aux_shared[idx]
                        compu_targetscore_aux_all[nidx] = compu_targetscore_aux

                        X_aux_all = torch.empty([Z_aux.shape[0], Z_aux.shape[1], X_aux_shared.shape[2]], device = Z_aux.device)
                        X_aux_all[idx]  = X_aux_shared[idx]
                        X_aux_all[nidx] = X_aux


                        log_kxy = log_gaussian_kernel(X.detach(), X_aux_all)
                        log_w = std_normal.log_prob(Z_aux).sum(-1) - lps + log_kxy
                        ker_score_diff = ((-neg_score_implicit_aux_all -  compu_targetscore_aux_all) * log_w[..., None].exp()).mean(0)


                    loss_kxy = (ker_score_diff * X).sum(dim=1).mean()


                    if epoch < 0:
                        loss_logp = (self.target_model.logp(X)).mean()
                        loss_kxy = loss_kxy - loss_logp
                    optimizer_VI.zero_grad()
                    loss_kxy.backward()
                    optimizer_VI.step()
                    scheduler_VI.step()

                    #print("Time:", time.time() - start)
                    #exit()


                # compute some object in the trainging


                with torch.no_grad():
                    Z = torch.randn([60000, config.train.z_dim]).to(self.device)

                    samples, _ = self.SemiVInet(Z)

                    dist = torch.distributions.Normal(samples[None, ...], 0.01)
                    sgld_samples_t = sgld_samples[:, None, ...]
                lp = None

                for i in range(10):
                    if lp is None:
                        lp = (dist.log_prob(sgld_samples_t[i*100:(1+i)*100]).logsumexp(dim=1) - torch.tensor(samples.shape[0]).log()).sum()
                    else:
                        lp += (dist.log_prob(sgld_samples_t[i*100:(1+i)*100]).logsumexp(dim=1) - torch.tensor(samples.shape[0]).log()).sum()
                        #lp += (dist.log_prob(sgld_samples_t[i*100:(1+i)*100]).logsumexp(dim=1) - torch.tensor(samples.shape[0]).log()).sum()

                mll.append(lp)
                loss_list.append(np.array([self.iter_idx, loss_kxy.item()]))    
                logger.info(("Epoch [{}/{}], iters [{}], loss: {:.4f}, net_log_var: {:.4f}").format(epoch, self.trainpara.num_epochs, self.iter_idx, loss_kxy, self.SemiVInet.log_var.mean().item()))
                if epoch%self.config.sampling.visual_time ==0:
                    X = self.SemiVInet.sampling(num = self.config.sampling.num)
                    figname = str(self.iter_idx+1).zfill(8) + '.jpg'
                    self.target_model.trace_plot(X, figpath = self.save_samples_to_path, figname = figname)

            loss_list = np.array(loss_list)
            X = self.SemiVInet.sampling(num = self.config.sampling.num)
            torch.save(X.cpu().numpy(), os.path.join(self.save_samples_to_path,str(k) + 'sample{}.pt'.format(self.config.sampling.num)))
            torch.save(loss_list, os.path.join(self.config.log_path, str(k) + 'loss_list.pt'))
            torch.save(self.SemiVInet.state_dict(), os.path.join(self.config.log_path, str(k) + "SemiVInet.ckpt"))
            torch.save(mll, os.path.join(self.config.log_path,str(k) + "loss.pt"))
            
        return loss_list

if __name__ == "__main__":
    seednow = 2023
    torch.manual_seed(seednow)
    torch.cuda.manual_seed_all(seednow)
    np.random.seed(seednow)
    random.seed(seednow)
    torch.backends.cudnn.deterministic = True

    config = parse_config()

    datetimelabel = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') if config.log_stick else "Now_vanilla"
    config.log_path = os.path.join("pathexpkernel-isSIVI", config.target_score, "{}".format(datetimelabel))
    os.makedirs(config.log_path,exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    filehandler = logging.FileHandler(os.path.join(config.log_path,"final.log"))
    filehandler.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.info('Training with the following settings:')
    for name, value in vars(config).items():
        if isinstance(value, argparse.Namespace):
            for name, value in vars(value).items():
                logger.info('{} : {}'.format(name, value))
        else:
            logger.info('{} : {}'.format(name, value))
    config.logger = logger
    task = SIVIPathsteinISLangevin(config)
    task.learn()
