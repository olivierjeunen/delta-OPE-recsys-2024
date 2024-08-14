import argparse, sys, os, pickle, random
from pathlib import Path
import itertools

from joblib import delayed
from joblib import Parallel
import multiprocessing
import numpy as np
from pandas import DataFrame
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import yaml, gc

sys.path.append('./')
sys.path.append('./examples')

import scipy
import matplotlib.pyplot as plt 

from obp.dataset import OpenBanditDataset
from obp.dataset import (
    SyntheticBanditDataset,
    logistic_reward_function,
    logistic_sparse_reward_function,
    linear_behavior_policy
)
from obp.ope import DirectMethod
from obp.ope import DoublyRobust
from obp.ope import DoublyRobustWithShrinkageTuning
from obp.ope import InverseProbabilityWeighting
from obp.ope import OffPolicyEvaluation
from obp.ope import RegressionModel
from obp.ope import SelfNormalizedDoublyRobust
from obp.ope import SelfNormalizedInverseProbabilityWeighting
from obp.ope import InverseProbabilityWeightingOptimalBaseline
from obp.ope import SwitchDoublyRobustTuning
from obp.policy import BernoulliTS
from obp.policy import Random
from obp.policy import IPWLearner

from tqdm.notebook import tqdm



evaluation_policy_dict = dict(bts=BernoulliTS, random=Random)
base_path = '<ABSOLUTE_PATH_TO_CODE_HOMEFOLDER>'
log_path = '<ABSOLUTE_PATH_TO_CODE_HOMEFOLDER>/logs/ope'

# hyperparameters of the regression model used in model dependent OPE estimators
with open(os.path.join(base_path, "examples/obd/conf/hyperparams.yaml"), "rb") as f:
    hyperparams = yaml.safe_load(f)

base_model_dict = dict(
    logistic_regression=LogisticRegression,
    lightgbm=GradientBoostingClassifier,
    random_forest=RandomForestClassifier,
)

# compared OPE estimators
ope_estimators = [
    DirectMethod(),
    InverseProbabilityWeighting(),
    SelfNormalizedInverseProbabilityWeighting(),
    DoublyRobust(),
    InverseProbabilityWeightingOptimalBaseline(),
    SelfNormalizedDoublyRobust(),
    SwitchDoublyRobustTuning(lambdas=[10, 50, 100, 500, 1000, 5000, 10000, np.inf]),
    DoublyRobustWithShrinkageTuning(
        lambdas=[10, 50, 100, 500, 1000, 5000, 10000, np.inf]
    ),
]

estimator_dicts = {'ipw': 'IPS', 'dm': 'DM', 'dr': 'DR', 'ipwob': 'beta-IPS', 'snipw': 'SNIPS'}

print(ope_estimators)

def ope_round(i, N, n_actions):
    method2samplesize2err = {
    'IPS': {},    
    'SNIPS': {},  
    'DM': { },  
    'DR': {},
    'beta-IPS': {}
    }
    # The setup used for the RecSys experiments
    dataset = SyntheticBanditDataset(
                n_actions=n_actions,
                dim_context=5,
                beta=beta, # inverse temperature parameter to control the optimality and entropy of the behavior policy
                reward_type="binary", # "binary" or "continuous"
                reward_function=logistic_sparse_reward_function,
                #behavior_policy_function=linear_behavior_policy
                random_state=i,
                )
    # The original setup from the OBP documentation.
    # dataset = SyntheticBanditDataset(
    #             n_actions=n_actions,
    #             dim_context=5,
    #             beta=1, # inverse temperature parameter to control the optimality and entropy of the behavior policy
    #             reward_type="binary", # "binary" or "continuous"
    #             reward_function=logistic_reward_function,
    #             #behavior_policy_function=linear_behavior_policy
    #             random_state=i,
    #             )
    bandit_feedback_train = dataset.obtain_batch_bandit_feedback(n_rounds=int(N))
    bandit_feedback_test = dataset.obtain_batch_bandit_feedback(n_rounds=int(N)) 
            
    ipw_lr = IPWLearner(
            n_actions=dataset.n_actions,
            base_classifier=LogisticRegression()
            )

    # train IPWLearner on the training set of logged bandit data
    ipw_lr.fit(
        context=bandit_feedback_train["context"],
        action=bandit_feedback_train["action"],
        reward=bandit_feedback_train["reward"],
        pscore=bandit_feedback_train["pscore"]
    )
    action_dist_ipw_lr = ipw_lr.predict(context=bandit_feedback_test["context"])

    # estimate the expected rewards by using an ML model (Logistic Regression here)
    # the estimated rewards are used by model-dependent estimators such as DM and DR
    regression_model = RegressionModel(
        n_actions=dataset.n_actions,
        action_context=dataset.action_context,
        base_model=LogisticRegression(),
    )
    # estimated_rewards_by_reg_model = regression_model.fit_predict(
    #                         context=bandit_feedback_test["context"],
    #                         action=bandit_feedback_test["action"],
    #                         reward=bandit_feedback_test["reward"],
    #                         n_folds=3, # use 3-fold cross-fitting
    #                         )
    regression_model.fit(
                            context=bandit_feedback_train["context"],
                            action=bandit_feedback_train["action"],
                            reward=bandit_feedback_train["reward"]
                            )
    estimated_rewards_by_reg_model = regression_model.predict(
                            context=bandit_feedback_test["context"]
                            )
    # estimate the policy value of the evaluation policies based on their action choice probabilities
    # it is possible to set multiple OPE estimators to the `ope_estimators` argument
    ope = OffPolicyEvaluation(
        bandit_feedback=bandit_feedback_test,
        ope_estimators=[InverseProbabilityWeighting(), DirectMethod(), DoublyRobust(), InverseProbabilityWeightingOptimalBaseline(), SelfNormalizedInverseProbabilityWeighting()]
    )
    relative_ee_for_ipw_lr = ope.summarize_estimators_comparison(
        ground_truth_policy_value=dataset.calc_ground_truth_policy_value(
            expected_reward=bandit_feedback_test["expected_reward"],
            action_dist=action_dist_ipw_lr,
        ),
        action_dist=action_dist_ipw_lr,
        estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        metric="se", # "relative-ee" (relative estimation error) or "se" (squared error)
        )
    estimated_policy_value_a, estimated_interval_a, estimated_sample_variance = ope.summarize_off_policy_estimates(
    action_dist=action_dist_ipw_lr,
    estimated_rewards_by_reg_model=estimated_rewards_by_reg_model
    )
    print(estimated_sample_variance)
    mse_values = relative_ee_for_ipw_lr.values.flatten().tolist() 
    policy_values = estimated_sample_variance['sample_variance'].tolist()
    
    # first value is the MSE, second one is the estimated policy value
    method2samplesize2err['IPS'] = (mse_values[0], policy_values[0]/N)
    method2samplesize2err['DM'] = (mse_values[1], policy_values[1]/N)
    method2samplesize2err['DR'] = (mse_values[2], policy_values[2]/N)
    method2samplesize2err['beta-IPS'] = (mse_values[3], policy_values[3]/N)
    method2samplesize2err['SNIPS'] = (mse_values[4], policy_values[4])
    #print(method2samplesize2err)
    #gc.collect()
    return method2samplesize2err


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate off-policy estimators.")
    
    
    parser.add_argument(
        "--beta",
        type=float,
        default=1,
        help="Temperature param for softmax for behavior policy.",
    )
    
    parser.add_argument(
        "--N",
        type=float,
        default=1000,
        help="Number of points in the logged data.",
    )
    
    
    parser.add_argument(
        "--iteration",
        type=int,
        default=1,
        help="Iteration number",
    )
    
    parser.add_argument("--random_state", type=int, default=12345)
    args = parser.parse_args()
    print(args)

    # configurations
    n_actions = [25, 50, 75, 100]
    beta = args.beta
    random_state = args.random_state
    num_processes = multiprocessing.cpu_count()  # You can adjust this number as needed
    print("number of processes %d"%(num_processes))
    #np.random.seed(random_state)
    
    for n_action in n_actions:
        method2samplesize2err = ope_round(args.iteration, args.N, n_action)
        with open(os.path.join(log_path, 'ope_' + str(args.iteration) + '_' + str(n_action) + '_' + str(args.N) + '_' + str(beta)), 'wb') as fp:
            pickle.dump(method2samplesize2err, fp)
        
                
       
    
    
