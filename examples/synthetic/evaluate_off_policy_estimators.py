import argparse
from pathlib import Path

from joblib import delayed
from joblib import Parallel
import numpy as np
from pandas import DataFrame
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import yaml

from obp.dataset import logistic_reward_function
from obp.dataset import SyntheticBanditDataset
from obp.ope import DirectMethod
from obp.ope import DoublyRobust
from obp.ope import DoublyRobustWithShrinkageTuning
from obp.ope import InverseProbabilityWeighting
from obp.ope import OffPolicyEvaluation
from obp.ope import RegressionModel
from obp.ope import SelfNormalizedDoublyRobust
from obp.ope import SelfNormalizedInverseProbabilityWeighting
from obp.ope import SwitchDoublyRobustTuning
from obp.policy import IPWLearner


# hyperparameters of the regression model used in model dependent OPE estimators
with open("./conf/hyperparams.yaml", "rb") as f:
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
    SelfNormalizedDoublyRobust(),
    SwitchDoublyRobustTuning(lambdas=[10, 50, 100, 500, 1000, 5000, 10000, np.inf]),
    DoublyRobustWithShrinkageTuning(
        lambdas=[10, 50, 100, 500, 1000, 5000, 10000, np.inf]
    ),
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="evaluate the accuracy of OPE estimators on synthetic bandit data."
    )
    parser.add_argument(
        "--n_runs", type=int, default=1, help="number of simulations in the experiment."
    )
    parser.add_argument(
        "--n_rounds",
        type=int,
        default=10000,
        help="sample size of logged bandit data.",
    )
    parser.add_argument(
        "--n_actions",
        type=int,
        default=10,
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
        default=3,
        help="inverse temperature parameter to control the behavior policy.",
    )
    parser.add_argument(
        "--base_model_for_evaluation_policy",
        type=str,
        choices=["logistic_regression", "lightgbm", "random_forest"],
        required=True,
        help="base ML model for evaluation policy, logistic_regression, random_forest or lightgbm.",
    )
    parser.add_argument(
        "--base_model_for_reg_model",
        type=str,
        choices=["logistic_regression", "lightgbm", "random_forest"],
        required=True,
        help="base ML model for regression model, logistic_regression, random_forest or lightgbm.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="the maximum number of concurrently running jobs.",
    )
    parser.add_argument("--random_state", type=int, default=12345)
    args = parser.parse_args()
    print(args)

    # configurations
    n_runs = args.n_runs
    n_rounds = args.n_rounds
    n_actions = args.n_actions
    dim_context = args.dim_context
    beta = args.beta
    base_model_for_evaluation_policy = args.base_model_for_evaluation_policy
    base_model_for_reg_model = args.base_model_for_reg_model
    n_jobs = args.n_jobs
    random_state = args.random_state

    def process(i: int):
        # synthetic data generator
        dataset = SyntheticBanditDataset(
            n_actions=n_actions,
            dim_context=dim_context,
            reward_function=logistic_reward_function,
            beta=beta,
            random_state=i,
        )
        # define evaluation policy using IPWLearner
        evaluation_policy = IPWLearner(
            n_actions=dataset.n_actions,
            base_classifier=base_model_dict[base_model_for_evaluation_policy](
                **hyperparams[base_model_for_evaluation_policy]
            ),
        )
        # sample new training and test sets of synthetic logged bandit data
        bandit_feedback_train = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
        bandit_feedback_test = dataset.obtain_batch_bandit_feedback(n_rounds=n_rounds)
        # train the evaluation policy on the training set of the synthetic logged bandit data
        evaluation_policy.fit(
            context=bandit_feedback_train["context"],
            action=bandit_feedback_train["action"],
            reward=bandit_feedback_train["reward"],
            pscore=bandit_feedback_train["pscore"],
        )
        # predict the action decisions for the test set of the synthetic logged bandit data
        action_dist = evaluation_policy.predict_proba(
            context=bandit_feedback_test["context"],
        )
        # estimate the reward function of the test set of synthetic bandit feedback with ML model
        regression_model = RegressionModel(
            n_actions=dataset.n_actions,
            action_context=dataset.action_context,
            base_model=base_model_dict[base_model_for_reg_model](
                **hyperparams[base_model_for_reg_model]
            ),
        )
        estimated_rewards_by_reg_model = regression_model.fit_predict(
            context=bandit_feedback_test["context"],
            action=bandit_feedback_test["action"],
            reward=bandit_feedback_test["reward"],
            n_folds=3,  # 3-fold cross-fitting
            random_state=random_state,
        )
        # evaluate estimators' performances using relative estimation error (relative-ee)
        ope = OffPolicyEvaluation(
            bandit_feedback=bandit_feedback_test,
            ope_estimators=ope_estimators,
        )
        metric_i = ope.evaluate_performance_of_estimators(
            ground_truth_policy_value=dataset.calc_ground_truth_policy_value(
                expected_reward=bandit_feedback_test["expected_reward"],
                action_dist=action_dist,
            ),
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            metric="relative-ee",
        )

        return metric_i

    processed = Parallel(
        n_jobs=n_jobs,
        verbose=50,
    )([delayed(process)(i) for i in np.arange(n_runs)])
    metric_dict = {est.estimator_name: dict() for est in ope_estimators}
    for i, metric_i in enumerate(processed):
        for (
            estimator_name,
            relative_ee_,
        ) in metric_i.items():
            metric_dict[estimator_name][i] = relative_ee_
    results_df = DataFrame(metric_dict).describe().T.round(6)

    print("=" * 45)
    print(f"random_state={random_state}")
    print("-" * 45)
    print(results_df[["mean", "std"]])
    print("=" * 45)

    # save results of the evaluation of OPE in './logs' directory.
    log_path = Path("./logs")
    log_path.mkdir(exist_ok=True, parents=True)
    results_df.to_csv(log_path / "evaluation_of_ope_results.csv")
