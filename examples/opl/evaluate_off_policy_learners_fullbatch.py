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
                        train_ips_opt_fb, train_snips

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
log_path = '<ABSOLUTE_PATH_TO_CODE_HOMEFOLDER>/logs/opl/fb'


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
        default=-5,
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
    random_state = args.random_state 
    random_state_dataset = 12345
    torch.manual_seed(random_state)
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
        random_state=random_state_dataset,
    )
    # sample new training and test sets of synthetic logged bandit data
    bandit_feedback_train = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
    bandit_feedback_test = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)

    # define random evaluation policy
    random_policy = Random(n_actions=dataset.n_actions, random_state=random_state)
    # predict the action decisions for the test set of the synthetic logged bandit data
    random_action_dist = random_policy.compute_batch_action_dist(n_rounds=n_rounds)
    
    context_train=bandit_feedback_train["context"]
    action_train=bandit_feedback_train["action"]
    reward_train=bandit_feedback_train["reward"]
    pscore_train=bandit_feedback_train["pscore"]
        
    dataset_train_torch = CustomDataset(context_train,
                action_train,
                reward_train,
                pscore_train)
    
    bandit_train_loader = DataLoader(dataset_train_torch,
                                    batch_size=len(context_train), 
                                    shuffle=False)
    
    
    # train optimal baseline IPW policy
    hnet = PolicyNet(dim_context, n_actions)
    if optimizer == "adam":
        opt = torch.optim.Adam(hnet.parameters(), lr=lr, weight_decay=0.0)
    elif optimizer == "adamW":
        opt = torch.optim.AdamW(hnet.parameters(), lr=lr)
    else:
        opt = torch.optim.SGD(hnet.parameters(), lr=lr) 
           
    policy_values_opt_baseline, grad_vars_opt_baseline = train_ips_opt_fb(n_epochs, bandit_train_loader, hnet, bandit_feedback_test, dataset, device, opt)
    with open(os.path.join(log_path, 'opt_baseline_val_' + optimizer + '_' + str(lr) + '_' + str(pid)), 'wb') as fp:
        pickle.dump(policy_values_opt_baseline, fp)
    
    with open(os.path.join(log_path, 'opt_baseline_grad_var_' + optimizer + '_' + str(lr) + '_' + str(pid)), 'wb') as fp:
        pickle.dump(grad_vars_opt_baseline, fp)
        
    ipw_opt_baseline_learner_action_dist = predict(hnet, device,
        context=bandit_feedback_test["context"],
    )
    
    hnet = PolicyNet(dim_context, n_actions)
    if optimizer == "adam":
        opt = torch.optim.Adam(hnet.parameters(), lr=lr, weight_decay=0.0)
    elif optimizer == "adamW":
        opt = torch.optim.AdamW(hnet.parameters(), lr=lr)
    else:
        opt = torch.optim.SGD(hnet.parameters(), lr=lr) 
    policy_values_ips, grad_vars_ips = train_ips(n_epochs, bandit_train_loader, hnet, bandit_feedback_test, dataset, device, opt)
    with open(os.path.join(log_path, 'ips_val_' + optimizer + '_' + str(lr) + '_' + str(pid)), 'wb') as fp:
        pickle.dump(policy_values_ips, fp)
        
    with open(os.path.join(log_path, 'ips_grad_var_' + optimizer + '_' + str(lr) + '_' + str(pid)), 'wb') as fp:
        pickle.dump(grad_vars_ips, fp)
        
    ipw_learner_action_dist = predict(hnet,
                                      device,
                                    context=bandit_feedback_test["context"]
    )
    
    hnet = PolicyNet(dim_context, n_actions)
    if optimizer == "adam":
        opt = torch.optim.Adam(hnet.parameters(), lr=lr, weight_decay=0.0)
    elif optimizer == "adamW":
        opt = torch.optim.AdamW(hnet.parameters(), lr=lr)
    else:
        opt = torch.optim.SGD(hnet.parameters(), lr=lr) 
    
    policy_value_banditnet, grad_vars_banditnet = train_snips(n_epochs, bandit_train_loader, hnet, bandit_feedback_test, dataset, device, opt )
    with open(os.path.join(log_path, 'banditnet_val_' + optimizer + '_' + str(lr) + '_' + str(pid)), 'wb') as fp:
        pickle.dump(policy_value_banditnet, fp)
        
    with open(os.path.join(log_path, 'banditnet_grad_var_' + optimizer + '_' + str(lr) + '_' + str(pid)), 'wb') as fp:
        pickle.dump(grad_vars_banditnet, fp) 
        
    snips_learner_action_dist = predict(hnet, device,
        context=bandit_feedback_test["context"],
    )
    print("IPS policy values in test")
    print(policy_values_ips)
    print("BanditNet policy values in test")
    print(policy_value_banditnet)
    print("IPS optimal baseline values in test")
    print(policy_values_opt_baseline)
    # The code below is from the OBP code, might be useful for debugging
    random_policy_value = dataset.calc_ground_truth_policy_value(
        expected_reward=bandit_feedback_test["expected_reward"],
        action_dist=random_action_dist,
    )
    ipw_learner_policy_value = dataset.calc_ground_truth_policy_value(
        expected_reward=bandit_feedback_test["expected_reward"],
        action_dist=ipw_learner_action_dist,
    )
    snips_learner_policy_value = dataset.calc_ground_truth_policy_value(
        expected_reward=bandit_feedback_test["expected_reward"],
        action_dist=snips_learner_action_dist,
    )
    opt_baseline_policy_value = dataset.calc_ground_truth_policy_value(
        expected_reward=bandit_feedback_test["expected_reward"],
        action_dist=ipw_opt_baseline_learner_action_dist,
    )

    policy_value_df = DataFrame(
        [
            [random_policy_value],
            [ipw_learner_policy_value],
            [snips_learner_policy_value],
            [opt_baseline_policy_value]
        ],
        columns=["policy value"],
        index=[
            "random_policy",
            "ipw_learner",
            "BanditNet learning",
            "ipw with optimal baseline learner"
        ],
    ).round(6)
    print("=" * 45)
    print(f"random_state={random_state}")
    print("-" * 45)
    print(policy_value_df)
    print("=" * 45)

    # save results of the evaluation of off-policy learners in './logs' directory.
    log_path = Path("./logs")
    log_path.mkdir(exist_ok=True, parents=True)
    policy_value_df.to_csv(log_path / "policy_value_of_off_policy_learners.csv")
    plot_performance(policy_values_ips, policy_value_banditnet, policy_values_opt_baseline)
    