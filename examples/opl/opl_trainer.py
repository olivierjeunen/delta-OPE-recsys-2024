import argparse, os, sys
from pathlib import Path

from pandas import DataFrame
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import yaml
from collections import defaultdict

sys.path.append('./')
sys.path.append('./examples')


from obp.dataset import logistic_reward_function
from obp.dataset import SyntheticBanditDataset
from obp.policy import IPWLearner
from obp.policy import NNPolicyLearner
from obp.policy import Random

from sklearn.ensemble import RandomForestClassifier as RandomForest
from sklearn.linear_model import LogisticRegression
from functorch import make_functional_with_buffers, vmap, grad
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim.lr_scheduler as lr_scheduler

import pdb
import numpy as np


#torch.manual_seed(0)
class CustomDataset(Dataset):
    def __init__(self, context, action, reward, pscore ):
        self.context = context
        self.action = action
        self.reward = reward
        self.pscore = pscore

    def __len__(self):
        return len(self.context)

    def __getitem__(self, idx):
        X = self.context[idx, :]
        s_labels = self.action[idx]
        s_log_prop = self.pscore[idx]
        s_loss = self.reward[idx]
        y = self.action[idx]

        return X, s_labels, s_log_prop, s_loss, y

class PolicyNet(nn.Module):
    
    def __init__(self, context_dim, n_actions, device='cuda:0'):
        super().__init__()
        # For now trying out a linear policy with bias.
        self.fc1 = nn.Linear(context_dim, n_actions, bias=False)
        self.n_actions = n_actions
        self.device = device

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.softmax(x,dim=-1)
        return x
    
    def predict(self, context):
        x = torch.tensor(context).float().to(self.device)
        y = self.forward(x).detach().cpu().numpy()
        n = context.shape[0]
        predicted_actions = np.argmax(y, axis=1)
        action_dist = np.zeros((n, self.n_actions, 1))
        action_dist[np.arange(n), predicted_actions, 0] = 1
        return action_dist

    
# Helper functions to calculate the per-sample gradients
def loss_fn(predictions, targets):
    # pdb.set_trace()
    bs = predictions.shape[0]
    return predictions[:, targets].flatten().sum()

def compute_loss_stateless_model(params, buffers, sample, target, fmodel):
    batch = sample.unsqueeze(0)
    targets = target.unsqueeze(0)

    predictions = fmodel(params, buffers, batch)
    loss = loss_fn(predictions, targets)
    return loss


def train_ips_optimal(max_epoch, bandit_train_loader, hnet, bandit_feedback_test, dataset, device, opt):
    ft_compute_grad = grad(compute_loss_stateless_model)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=( None, None,0, 0, None))
    hnet.train()
    hnet = hnet.to(device)
    #weight_grad_baseline = torch.tensor(0.).float().to(device)
    N = 1
    policy_values = []
    weight_grad_var = []
    N_running_average = 1
    for epoch in range(max_epoch):
        weight_grad_var_batch = 0.
        N = 0
        for ele in bandit_train_loader:
            X, a, p, r, _ = ele
            X = X.to(device)
            a = a.to(device)
            p = p.to(device)
            r = r.to(device)
            bs = X.shape[0]
            prob = hnet(X.float())
            pi = prob[range(bs), a]
            iw = pi/p
            # calculate optimal baseline
            fmodel, params, buffers = make_functional_with_buffers(hnet)
            batch_grads = ft_compute_sample_grad(params, buffers, X.to(torch.float32), a, fmodel)
            weight_grad = batch_grads[0].detach()
            weight_grad_flat = weight_grad.reshape(bs, -1)
            weight_grad_norm = torch.norm(weight_grad_flat, dim=-1)
            weight_grad_baseline = (torch.mean((weight_grad_norm/p)**2 * r) / torch.mean((weight_grad_norm/p)**2))
            #weight_grad_baseline/= N_running_average
            #N_running_average += 1
            # sample gradient variance for the mini-batch
            # Second part of the variance formula (Eq. 28 in the draft)
            temp2 = (weight_grad_flat/p.unsqueeze(1)) * (r - weight_grad_baseline).unsqueeze(1)
            temp2 = torch.mean(torch.norm(temp2.mean(0))**2)
            # first part of the variance formula
            temp1 = weight_grad_norm**2 * ((r-weight_grad_baseline)/p)**2
            temp1 = torch.mean(temp1)
            batch_grad_var = (temp1 - temp2)/bs
            weight_grad_var_batch += (batch_grad_var.item())
            N += bs
            loss_weight = -(iw * (r - weight_grad_baseline)).mean()
            loss_weight.backward()
            opt.step()
            opt.zero_grad()
        learner_action_dist = hnet.predict( 
        context=bandit_feedback_test["context"],
        )
        policy_value = dataset.calc_ground_truth_policy_value(
        expected_reward=bandit_feedback_test["expected_reward"],
        action_dist=learner_action_dist,
        )
        policy_values.append(policy_value)
        weight_grad_var.append(weight_grad_var_batch/N)
        #scheduler.step()
    return policy_values, weight_grad_var
            

def train_ips_banditnet(max_epoch, bandit_train_loader, hnet, bandit_feedback_test, dataset, device, opt, banditnet_lambda=None):
    ft_compute_grad = grad(compute_loss_stateless_model)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=( None, None,0, 0, None))
    #scheduler = lr_scheduler.LinearLR(opt, start_factor=0.1, end_factor=1e-4, total_iters=50)
    hnet.train()
    hnet = hnet.to(device)
    policy_values = []
    weight_grad_var = []
    for epoch in range(max_epoch):
        weight_grad_var_batch = 0.
        N = 0
        for ele in bandit_train_loader:
            X, a, p, r, _ = ele
            X = X.to(device)
            a = a.to(device)
            p = p.to(device)
            r = r.to(device)
            bs = X.shape[0]
            prob = hnet(X.float())
            pi = prob[range(bs), a]
            iw = pi/p
            # calculate individual gradients
            fmodel, params, buffers = make_functional_with_buffers(hnet)
            batch_grads = ft_compute_sample_grad(params, buffers, X.to(torch.float32), a, fmodel)
            weight_grad = batch_grads[0].detach()
            weight_grad_flat = weight_grad.reshape(bs, -1)
            weight_grad_norm = torch.norm(weight_grad_flat, dim=-1)
            # sample gradient variance for the mini-batch
            # Second part of the variance formula (Eq. 28 in the draft)
            temp2 = (weight_grad_flat/p.unsqueeze(1)) * (r -banditnet_lambda).unsqueeze(1)
            temp2 = torch.mean(torch.norm(temp2.mean(0))**2)
            # first part of the variance formula
            temp1 = weight_grad_norm**2 * ((r-banditnet_lambda)/p)**2
            temp1 = torch.mean(temp1)
            batch_grad_var = (temp1 - temp2)/bs
            weight_grad_var_batch += batch_grad_var.item()
            N += bs
            loss = -iw * (r - banditnet_lambda)
            loss = loss.mean()
            loss.backward()
            opt.step()
            opt.zero_grad()
        learner_action_dist = hnet.predict( 
        context=bandit_feedback_test["context"],
        )
        policy_value = dataset.calc_ground_truth_policy_value(
        expected_reward=bandit_feedback_test["expected_reward"],
        action_dist=learner_action_dist,
        )
        policy_values.append(policy_value)
        weight_grad_var.append(weight_grad_var_batch/N)
        #scheduler.step()
    return policy_values, weight_grad_var
 
 
def eval_banditnet(bandit_train_loader, hnet, device):
    ### evaluate SNIPS metric on the train set
    banditnet_num = torch.tensor(0.).to(device)
    banditnet_den = torch.tensor(0.).to(device)
    for ele in bandit_train_loader:
        X, a, p, r, _ = ele
        bs = X.shape[0]
        X = X.to(device)
        a = a.to(device)
        p = p.to(device)
        r = r.to(device)
        prob = hnet(X.float())
        pi = prob[range(bs), a]
        iw = pi/p
        banditnet_num += (iw * r).sum()
        banditnet_den += iw.sum()
    return banditnet_num/banditnet_den
        
def tune_lambda_banditnet(max_epoch, bandit_train_loader, bandit_feedback_train, bandit_feedback_test, dataset, dim_context, n_actions, device, optimizer, lr):
      # BanditNet tuning loop
    lambda_range = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    n_trails = 3
    ips_banditnet_perf = defaultdict(list)
    
    
    for Lambda in lambda_range:
        torch.manual_seed(int(Lambda*100))
        hnet = PolicyNet(dim_context, n_actions)
        if optimizer == "adam":
            opt = torch.optim.Adam(hnet.parameters(), lr=lr, weight_decay=0.0)
        elif optimizer == "adamW":
            opt = torch.optim.AdamW(hnet.parameters(), lr=lr)
        else:
            opt = torch.optim.SGD(hnet.parameters(), lr=lr) 
        _,_ = train_ips_banditnet(max_epoch//2, bandit_train_loader, hnet, bandit_feedback_test, dataset, device, opt, banditnet_lambda=Lambda)
        #hnet = hnet.to('cpu')
        #action_dist_ips_banditnet_torch = hnet(torch.tensor(bandit_feedback_train["context"]).float()).detach()
        #snips_estimate_train = torch.mean((action_dist_ips_banditnet_torch[:, torch.tensor(bandit_feedback_train["action"]) ] * torch.tensor(bandit_feedback_train["reward"]))/torch.tensor(bandit_feedback_train["pscore"]))/ torch.mean((action_dist_ips_banditnet_torch[:, torch.tensor(bandit_feedback_train["action"]) ])/torch.tensor(bandit_feedback_train["pscore"]))
        snips_estimate_train = eval_banditnet(bandit_train_loader, hnet, device)
        ips_banditnet_perf[Lambda].append(snips_estimate_train.to('cpu').detach().numpy())
        print('one trail run completed!')
    for key, value in ips_banditnet_perf.items():
        avg_value = sum(value) / len(value)
        ips_banditnet_perf[key] = avg_value
    return ips_banditnet_perf


def train_ips(max_epoch, bandit_train_loader, hnet, bandit_feedback_test, dataset, device, opt):
    ft_compute_grad = grad(compute_loss_stateless_model)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=( None, None,0, 0, None))
    hnet.train()
    hnet = hnet.to(device)
    policy_values = []
    weight_grad_var = []
    for epoch in range(max_epoch):
        weight_grad_var_batch = 0.
        N = 0
        for ele in bandit_train_loader:
            X, a, p, r, _ = ele
            X = X.to(device)
            a = a.to(device)
            p = p.to(device)
            r = r.to(device)
            bs = X.shape[0]
            prob = hnet(X.float())
            pi = prob[range(bs), a]
            iw = pi/p
            # calculate individual gradients
            fmodel, params, buffers = make_functional_with_buffers(hnet)
            batch_grads = ft_compute_sample_grad(params, buffers, X.to(torch.float32), a, fmodel)
            weight_grad = batch_grads[0].detach()
            weight_grad_flat = weight_grad.reshape(bs, -1)
            weight_grad_norm = torch.norm(weight_grad_flat, dim=-1)
            # sample gradient variance for the mini-batch
            # Second part of the variance formula (Eq. 28 in the draft)
            temp2 = (weight_grad_flat/p.unsqueeze(1)) * (r ).unsqueeze(1)
            temp2 = torch.mean(torch.norm(temp2.mean(0))**2)
            # first part of the variance formula
            temp1 = weight_grad_norm**2 * ((r)/p)**2
            temp1 = torch.mean(temp1)
            batch_grad_var = (temp1 - temp2)/bs
            weight_grad_var_batch += batch_grad_var.item()
            N += bs
            
            loss = -iw * (r)
            loss = loss.mean()
            loss.backward()
            opt.step()
            opt.zero_grad()
        learner_action_dist = hnet.predict( 
        context=bandit_feedback_test["context"],
        )
        policy_value = dataset.calc_ground_truth_policy_value(
        expected_reward=bandit_feedback_test["expected_reward"],
        action_dist=learner_action_dist,
        )
        policy_values.append(policy_value)
        weight_grad_var.append(weight_grad_var_batch/N)
        #scheduler.step()
    return policy_values, weight_grad_var


def train_ips_opt_fb(max_epoch, bandit_train_loader, hnet, bandit_feedback_test, dataset, device, opt):
    ft_compute_grad = grad(compute_loss_stateless_model)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=( None, None,0, 0, None))
    hnet.train()
    hnet = hnet.to(device)
    #weight_grad_baseline = torch.tensor(0.).float().to(device)
    N = 1
    policy_values = []
    weight_grad_var = []
    N_running_average = 1
    for epoch in range(max_epoch):
        weight_grad_var_batch = 0.
        N = 0
        for ele in bandit_train_loader:
            X, a, p, r, _ = ele
            X = X.to(device)
            a = a.to(device)
            p = p.to(device)
            r = r.to(device)
            bs = X.shape[0]
            prob = hnet(X.float())
            pi = prob[range(bs), a]
            iw = pi/p
            # calculate optimal baseline
            fmodel, params, buffers = make_functional_with_buffers(hnet)
            batch_grads = ft_compute_sample_grad(params, buffers, X.to(torch.float32), a, fmodel)
            weight_grad = batch_grads[0].detach()
            weight_grad_flat = weight_grad.reshape(bs, -1)
            weight_grad_norm = torch.norm(weight_grad_flat, dim=-1)
            weight_grad_baseline = (torch.mean((weight_grad_norm/p)**2 * r) / torch.mean((weight_grad_norm/p)**2))
            #weight_grad_baseline/= N_running_average
            #N_running_average += 1
            # sample gradient variance for the mini-batch
            # Second part of the variance formula (Eq. 28 in the draft)
            temp2 = (weight_grad_flat/p.unsqueeze(1)) * (r - weight_grad_baseline).unsqueeze(1)
            temp2 = torch.mean(torch.norm(temp2.mean(0))**2)
            # first part of the variance formula
            temp1 = weight_grad_norm**2 * ((r-weight_grad_baseline)/p)**2
            temp1 = torch.mean(temp1)
            batch_grad_var = (temp1 - temp2)/bs
            weight_grad_var_batch += (batch_grad_var.item())
            N += bs
            beta = ((iw**2 - iw) * r)/(iw**2 - iw).mean()
            loss = (iw * (r - beta.mean())) + beta.mean()
            loss = -loss.mean()
            loss.backward()
            opt.step()
            opt.zero_grad()
        learner_action_dist = hnet.predict( 
        context=bandit_feedback_test["context"],
        )
        policy_value = dataset.calc_ground_truth_policy_value(
        expected_reward=bandit_feedback_test["expected_reward"],
        action_dist=learner_action_dist,
        )
        policy_values.append(policy_value)
        weight_grad_var.append(weight_grad_var_batch/N)
        #scheduler.step()
    return policy_values, weight_grad_var
            

def train_snips(max_epoch, bandit_train_loader, hnet, bandit_feedback_test, dataset, device, opt):
    ft_compute_grad = grad(compute_loss_stateless_model)
    ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=( None, None,0, 0, None))
    hnet.train()
    hnet = hnet.to(device)
    policy_values = []
    weight_grad_var = []
    for epoch in range(max_epoch):
        weight_grad_var_batch = 0.
        N = 0
        for ele in bandit_train_loader:
            X, a, p, r, _ = ele
            X = X.to(device)
            a = a.to(device)
            p = p.to(device)
            r = r.to(device)
            bs = X.shape[0]
            prob = hnet(X.float())
            pi = prob[range(bs), a]
            iw = pi/p
            # calculate individual gradients
            fmodel, params, buffers = make_functional_with_buffers(hnet)
            batch_grads = ft_compute_sample_grad(params, buffers, X.to(torch.float32), a, fmodel)
            weight_grad = batch_grads[0].detach()
            weight_grad_flat = weight_grad.reshape(bs, -1)
            weight_grad_norm = torch.norm(weight_grad_flat, dim=-1)
            # sample gradient variance for the mini-batch
            # Second part of the variance formula (Eq. 28 in the draft)
            temp2 = (weight_grad_flat/p.unsqueeze(1)) * (r ).unsqueeze(1)
            temp2 = torch.mean(torch.norm(temp2.mean(0))**2)
            # first part of the variance formula
            temp1 = weight_grad_norm**2 * ((r)/p)**2
            temp1 = torch.mean(temp1)
            batch_grad_var = (temp1 - temp2)/bs
            weight_grad_var_batch += batch_grad_var.item()
            N += bs
            
            loss = (iw * r)/iw.mean()
            loss = -loss.mean()
            loss.backward()
            opt.step()
            opt.zero_grad()
        learner_action_dist = hnet.predict( 
        context=bandit_feedback_test["context"],
        )
        policy_value = dataset.calc_ground_truth_policy_value(
        expected_reward=bandit_feedback_test["expected_reward"],
        action_dist=learner_action_dist,
        )
        policy_values.append(policy_value)
        weight_grad_var.append(weight_grad_var_batch/N)
        #scheduler.step()
    return policy_values, weight_grad_var