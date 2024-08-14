import argparse, os, sys, random, pickle
from pathlib import Path

from pandas import DataFrame
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import yaml
from torch.utils.data import DataLoader
import numpy as np

sys.path.append('./')
sys.path.append('./examples')

import torch

from obp.dataset import logistic_reward_function, logistic_sparse_reward_function
from obp.dataset import SyntheticBanditDataset
from obp.policy import IPWLearner
from obp.policy import NNPolicyLearner
from obp.policy import Random

import matplotlib.pyplot as plt 

from opl_trainer import PolicyNet, CustomDataset, train_ips,\
                        train_ips_banditnet, train_ips_optimal, tune_lambda_banditnet

# hyperparameters of the regression model used in model dependent OPE estimators
base_path = '<ABSOLUTE_PATH_TO_CODE_HOMEFOLDER>'

# hyperparameters of the regression model used in model dependent OPE estimators
with open(os.path.join(base_path, "examples/obd/conf/hyperparams.yaml"), "rb") as f:
    hyperparams = yaml.safe_load(f)


base_model_dict = dict(
    logistic_regression=LogisticRegression,
    lightgbm=GradientBoostingClassifier,
    random_forest=RandomForestClassifier,
)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

plot_path = '<ABSOLUTE_PATH_TO_CODE_HOMEFOLDER>/plots'
log_path = '<ABSOLUTE_PATH_TO_CODE_HOMEFOLDER>/logs/opl'


def plot_performance(policy_values_ips, policy_values_banditnet, policy_values_opt_baseline):
    fig, axes = plt.subplots(1, 1, figsize = (12, 3))
    epochs = range(len(policy_values_ips))
    policy_values = {}
    policy_values['IPS'] = policy_values_ips
    policy_values['BanditNet'] = policy_values_banditnet
    policy_values['BetaIPS'] = policy_values_opt_baseline
    
    for method, policy_values_method in policy_values.items():
        x = np.asarray(range(len(policy_values_method)))
        y = np.asarray(policy_values_method)
        
        print(method)
        print(y)
        axes.plot(x,y, label=method)
        #axes[0].fill_between(x, y-z*y_err, y+z*y_err, alpha=0.25)

        axes.set_xlabel('Training epoch')
        axes.set_ylabel('Policy Value')
        
        axes.grid(axis='y', ls='--', alpha=.25)
        #axes.set_yscale('symlog')
        axes.legend()
    plt.tight_layout()
    graph_path = os.path.join(plot_path, 'opl'  +'.pdf')
    plt.savefig(graph_path, bbox_inches='tight')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="evaluate off-policy estimators with synthetic bandit data."
    )
    parser.add_argument(
        "--n_rounds",
        type=int,
        default=20000,
        help="sample size of logged bandit data.",
    )
    parser.add_argument(
        "--n_actions",
        type=int,
        default=100,
        help="number of actions.",
    )
    parser.add_argument(
        "--dim_context",
        type=int,
        default=5,
        help="dimensions of context vectors.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=-4,
        help="inverse temperature parameter to control the behavior policy.",
    )
    
    parser.add_argument(
        "--optimizer", 
        type=str,
        default='sgd'
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int,
        default=256
    )
    
    parser.add_argument(
        "--learning_rate", 
        type=float,
        default=0.1, 
        help="learning rate for the optimizer"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int,
        default=100, 
        help="learning rate for the optimizer"
    )
    
    
    parser.add_argument("--random_state", type=int, default=12345)
    args = parser.parse_args()
    print(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = args.optimizer
    lr = args.learning_rate
    batch_size = args.batch_size
    # configurations
    n_rounds = args.n_rounds
    n_actions = args.n_actions
    dim_context = args.dim_context
    beta = args.beta
    pid = os.getpid()
    n_epochs = args.epochs

    random_state = args.random_state
    #torch.manual_seed(0)

    def predict(model, device, context):
        model.eval()
        
        x = torch.tensor(context).float().to(device)
        y = model(x).detach().to('cpu').numpy()
        n = context.shape[0]
        predicted_actions = np.argmax(y, axis=1)
        action_dist = np.zeros((n, n_actions, 1))
        action_dist[np.arange(n), predicted_actions, 0] = 1
        return action_dist

    # synthetic data generator
    dataset = SyntheticBanditDataset(
        n_actions=n_actions,
        dim_context=dim_context,
        reward_function=logistic_sparse_reward_function,
        beta=beta,
        random_state=random_state,
    )
    # sample new training and test sets of synthetic logged bandit data
    bandit_feedback_train = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
    bandit_feedback_test = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)

    context_train=bandit_feedback_train["context"]
    action_train=bandit_feedback_train["action"]
    reward_train=bandit_feedback_train["reward"]
    pscore_train=bandit_feedback_train["pscore"]
    
    context_test=bandit_feedback_test["context"]
    action_test=bandit_feedback_test["action"]
    reward_test=bandit_feedback_test["reward"]
    pscore_test=bandit_feedback_test["pscore"]
        
    dataset_train_torch = CustomDataset(context_train,
                action_train,
                reward_train,
                pscore_train)
    
    dataset_test_torch = CustomDataset(context_test,
                action_test,
                reward_test,
                pscore_test)
    
    bandit_train_loader = DataLoader(dataset_train_torch,
                                    batch_size=batch_size, 
                                    shuffle=True, num_workers=4)
    
    bandit_test_loader = DataLoader(dataset_test_torch,
                                    batch_size=batch_size, 
                                    shuffle=False)
    

    lambda_range = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    ips_banditnet_tune = tune_lambda_banditnet(n_epochs, bandit_train_loader, bandit_feedback_train, bandit_feedback_test, dataset, dim_context, n_actions, device, optimizer, lr)
    #ips_banditnet_tune = ips_banditnet_tune.to('cpu').detach()
    print(ips_banditnet_tune)
    best_lambda = max(ips_banditnet_tune, key=ips_banditnet_tune.get)

    print("best lambda for banditNet %f"%best_lambda)
       
    with open(os.path.join(log_path, 'banditnet_hyperparams'), 'wb') as fp:
        pickle.dump(ips_banditnet_tune, fp) 
        
    
    