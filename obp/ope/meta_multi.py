# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Off-Policy Evaluation Class to Streamline OPE with Multiple Loggers."""
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
import seaborn as sns
from sklearn.utils import check_scalar

from ..utils import check_array
from ..utils import check_confidence_interval_arguments
from .estimators import DirectMethod as DM
from .estimators_multi import MultiLoggersBalancedDoublyRobust as BalDR
from .estimators_multi import MultiLoggersNaiveDoublyRobust as NaiveDR
from .estimators_multi import MultiLoggersWeightedDoublyRobust as WeightedDR
from .meta import OffPolicyEvaluation


logger = getLogger(__name__)


@dataclass
class MultiLoggersOffPolicyEvaluation(OffPolicyEvaluation):
    """Class to conduct OPE with multiple loggers using multiple estimators simultaneously.

    Parameters
    -----------
    bandit_feedback: BanditFeedback
        Logged bandit data used to conduct OPE.

    ope_estimators: List[BaseMultiLoggersOffPolicyEstimator]
        List of OPE estimators used to evaluate the policy value of evaluation policy.
        Estimators must follow the interface of `BaseMultiLoggersOffPolicyEstimator`.

    Examples
    ----------

        .. code-block:: python

            # a case of implementing OPE (with multiple loggers) of an synthetic evaluation policy
            >>> from obp.dataset import (
                    SyntheticMultiLoggersBanditDataset,
                    logistic_reward_function,
                )
            >>> from obp.ope import (
                    MultiLoggersOffPolicyEvaluation,
                    MultiLoggersNaiveInverseProbabilityWeighting as NaiveIPW,
                    MultiLoggersBalancedInverseProbabilityWeighting as BalIPW,
                    MultiLoggersWeightedInverseProbabilityWeighting as WeightedIPW,
                )
            >>> from obp.utils import softmax

            # (1) Synthetic Data Generation
            >>> dataset = SyntheticMultiLoggersBanditDataset(
                    n_actions=10,
                    dim_context=5,
                    reward_function=logistic_reward_function,
                    betas=[-5, 0, 5],
                    rhos=[1, 1, 1],
                    random_state=12345,
                )
            >>> bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=10000)

            # (2) Synthetic Evaluation Policy
            >>> expected_reward = bandit_feedback["expected_reward"]
            >>> pi_e = softmax(10 * expected_reward)[:, :, np.newaxis]

            # (3) Off-Policy Evaluation
            >>> ope = MultiLoggersOffPolicyEvaluation(
                    bandit_feedback=bandit_feedback,
                    ope_estimators=[NaiveIPW(), BalIPW(), WeightedIPW()]
                )
            >>> estimated_policy_values = ope.estimate_policy_values(action_dist=pi_e)
            >>> estimated_policy_values

            {'multi_ipw': 0.735773315700805,
            'multi_bal_ipw': 0.7369963734186821,
            'multi_weighted_ipw': 0.7340662935948596}

            # (4) Ground-truth Policy Value of the Synthetic Evaluation Policy
            >>> dataset.calc_ground_truth_policy_value(
                    action_dist=pi_e, expected_reward=expected_reward,
                )

            0.7406153992278186

    """

    def __post_init__(self) -> None:
        """Initialize class."""
        for key_ in ["action", "position", "reward"]:
            if key_ not in self.bandit_feedback:
                raise RuntimeError(f"Missing key of {key_} in 'bandit_feedback'.")
        self.ope_estimators_ = dict()
        self.is_model_dependent = False
        for estimator in self.ope_estimators:
            self.ope_estimators_[estimator.estimator_name] = estimator
            if (
                isinstance(estimator, DM)
                or isinstance(estimator, NaiveDR)
                or isinstance(estimator, BalDR)
                or isinstance(estimator, WeightedDR)
            ):
                self.is_model_dependent = True

    def _create_estimator_inputs(
        self,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: Optional[
            Union[np.ndarray, Dict[str, np.ndarray]]
        ] = None,
        estimated_pscore: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        estimated_pscore_avg: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Create input dictionary to estimate policy value using subclasses of `BaseOffPolicyEstimator`"""
        check_array(array=action_dist, name="action_dist", expected_dim=3)
        if estimated_rewards_by_reg_model is None:
            pass
        elif isinstance(estimated_rewards_by_reg_model, dict):
            for estimator_name, value in estimated_rewards_by_reg_model.items():
                check_array(
                    array=value,
                    name=f"estimated_rewards_by_reg_model[{estimator_name}]",
                    expected_dim=3,
                )
                if value.shape != action_dist.shape:
                    raise ValueError(
                        f"Expected `estimated_rewards_by_reg_model[{estimator_name}].shape == action_dist.shape`, but found it False."
                    )
        else:
            check_array(
                array=estimated_rewards_by_reg_model,
                name="estimated_rewards_by_reg_model",
                expected_dim=3,
            )
            if estimated_rewards_by_reg_model.shape != action_dist.shape:
                raise ValueError(
                    "Expected `estimated_rewards_by_reg_model.shape == action_dist.shape`, but found it False"
                )
        for var_name, value_or_dict in {
            "estimated_pscore": estimated_pscore,
            "estimated_pscore_avg": estimated_pscore_avg,
        }.items():
            if value_or_dict is None:
                pass
            elif isinstance(value_or_dict, dict):
                for estimator_name, value in value_or_dict.items():
                    check_array(
                        array=value,
                        name=f"{var_name}[{estimator_name}]",
                        expected_dim=1,
                    )
                    if value.shape[0] != action_dist.shape[0]:
                        raise ValueError(
                            f"Expected `{var_name}[{estimator_name}].shape[0] == action_dist.shape[0]`, but found it False"
                        )
            else:
                check_array(array=value_or_dict, name=var_name, expected_dim=1)
                if value_or_dict.shape[0] != action_dist.shape[0]:
                    raise ValueError(
                        f"Expected `{var_name}.shape[0] == action_dist.shape[0]`, but found it False"
                    )

        estimator_inputs = {
            estimator_name: {
                input_: self.bandit_feedback[input_]
                for input_ in ["reward", "action", "position"]
            }
            for estimator_name in self.ope_estimators_
        }
        for estimator_name in self.ope_estimators_:
            for input_ in ["stratum_idx", "pscore", "pscore_avg"]:
                if input_ in self.bandit_feedback:
                    estimator_inputs[estimator_name][input_] = self.bandit_feedback[
                        input_
                    ]
                else:
                    estimator_inputs[estimator_name][input_] = None
            estimator_inputs[estimator_name]["action_dist"] = action_dist
            estimator_inputs = self._preprocess_model_based_input(
                estimator_inputs=estimator_inputs,
                estimator_name=estimator_name,
                model_based_input={
                    "estimated_rewards_by_reg_model": estimated_rewards_by_reg_model,
                    "estimated_pscore": estimated_pscore,
                    "estimated_pscore_avg": estimated_pscore_avg,
                },
            )
        return estimator_inputs

    def estimate_policy_values(
        self,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: Optional[
            Union[np.ndarray, Dict[str, np.ndarray]]
        ] = None,
        estimated_pscore: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        estimated_pscore_avg: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
    ) -> Dict[str, float]:
        """Estimate the policy value of evaluation policy.

        Parameters
        ------------
        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list) or Dict[str, array-like], default=None
            Expected rewards given each round, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_i,a_i)`.
            When an array-like is given, all OPE estimators use it.
            When a dict with an estimator's name as its key is given, the corresponding value is used for the estimator.
            If None, model-dependent estimators such as DM and DR cannot be used.

        estimated_pscore: array-like, shape (n_rounds,), default=None
            Estimated behavior policy (propensity scores), i.e., :math:`\\hat{\\pi}_k(a_i|x_i)`.
            When an array-like is given, all OPE estimators use it.
            When a dict with an estimator's name as its key is given, the corresponding value is used for the estimator.

        estimated_pscore_avg: array-like, shape (n_rounds,), default=None
            Estimated average behavior policy (propensity scores), i.e., :math:`\\hat{\\pi}_{avg}(a_i|x_i)`.
            When an array-like is given, all OPE estimators use it.
            When a dict with an estimator's name as its key is given, the corresponding value is used for the estimator.

        Returns
        ----------
        policy_value_dict: Dict[str, float]
            Dictionary containing the policy values estimated by OPE estimators.

        """
        if self.is_model_dependent:
            if estimated_rewards_by_reg_model is None:
                raise ValueError(
                    "When model dependent estimators such as DM or DR are used, `estimated_rewards_by_reg_model` must be given"
                )

        policy_value_dict = dict()
        estimator_inputs = self._create_estimator_inputs(
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            estimated_pscore=estimated_pscore,
            estimated_pscore_avg=estimated_pscore_avg,
        )
        for estimator_name, estimator in self.ope_estimators_.items():
            policy_value_dict[estimator_name] = estimator.estimate_policy_value(
                **estimator_inputs[estimator_name]
            )
        return policy_value_dict

    def estimate_intervals(
        self,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: Optional[
            Union[np.ndarray, Dict[str, np.ndarray]]
        ] = None,
        estimated_pscore: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        estimated_pscore_avg: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Estimate confidence intervals of policy values using bootstrap.

        Parameters
        ------------
        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list) or Dict[str, array-like], default=None
            Estimated expected rewards given context, action, and position, i.e., :math:`\\hat{q}(x_i,a_i)`.
            When an array-like is given, all OPE estimators use it.
            When a dict with an estimator's name as its key is given, the corresponding value is used for the estimator.
            If None, model-dependent estimators such as DM and DR cannot be used.

        estimated_pscore: array-like, shape (n_rounds,), default=None
            Estimated behavior policy (propensity scores), i.e., :math:`\\hat{\\pi}_b(a_i|x_i)`.
            When an array-like is given, all OPE estimators use it.
            When a dict with an estimator's name as its key is given, the corresponding value is used for the estimator.

        estimated_pscore_avg: array-like, shape (n_rounds,), default=None
            Estimated average behavior policy (propensity scores), i.e., :math:`\\hat{\\pi}_{avg}(a_i|x_i)`.
            When an array-like is given, all OPE estimators use it.
            When a dict with an estimator's name as its key is given, the corresponding value is used for the estimator.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=100
            Number of resampling performed in bootstrap sampling.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        policy_value_interval_dict: Dict[str, Dict[str, float]]
            Dictionary containing confidence intervals of the estimated policy values.

        """
        if self.is_model_dependent:
            if estimated_rewards_by_reg_model is None:
                raise ValueError(
                    "When model dependent estimators such as DM or DR are used, `estimated_rewards_by_reg_model` must be given"
                )

        check_confidence_interval_arguments(
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )
        policy_value_interval_dict = dict()
        estimator_inputs = self._create_estimator_inputs(
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            estimated_pscore=estimated_pscore,
            estimated_pscore_avg=estimated_pscore_avg,
        )
        for estimator_name, estimator in self.ope_estimators_.items():
            policy_value_interval_dict[estimator_name] = estimator.estimate_interval(
                **estimator_inputs[estimator_name],
                alpha=alpha,
                n_bootstrap_samples=n_bootstrap_samples,
                random_state=random_state,
            )

        return policy_value_interval_dict

    def summarize_off_policy_estimates(
        self,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: Optional[
            Union[np.ndarray, Dict[str, np.ndarray]]
        ] = None,
        estimated_pscore: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        estimated_pscore_avg: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
    ) -> Tuple[DataFrame, DataFrame]:
        """Summarize policy values and their confidence intervals estimated by OPE estimators.

        Parameters
        ------------
        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list) or Dict[str, array-like], default=None
            Expected rewards given each round, action, and position estimated by regression model, i.e., :math:`\\hat{q}(x_i,a_i)`.
            When an array-like is given, all OPE estimators use it.
            When a dict with an estimator's name as its key is given, the corresponding value is used for the estimator.
            If None, model-dependent estimators such as DM and DR cannot be used.

        estimated_pscore: array-like, shape (n_rounds,), default=None
            Estimated behavior policy (propensity scores), i.e., :math:`\\hat{\\pi}_b(a_i|x_i)`.
            When an array-like is given, all OPE estimators use it.
            When a dict with an estimator's name as its key is given, the corresponding value is used for the estimator.

        estimated_pscore_avg: array-like, shape (n_rounds,), default=None
            Estimated average behavior policy (propensity scores), i.e., :math:`\\hat{\\pi}_{avg}(a_i|x_i)`.
            When an array-like is given, all OPE estimators use it.
            When a dict with an estimator's name as its key is given, the corresponding value is used for the estimator.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=100
            Number of resampling performed in bootstrap sampling.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        (policy_value_df, policy_value_interval_df): Tuple[DataFrame, DataFrame]
            Policy values and their confidence intervals estimated by OPE estimators.

        """
        policy_value_df = DataFrame(
            self.estimate_policy_values(
                action_dist=action_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                estimated_pscore=estimated_pscore,
                estimated_pscore_avg=estimated_pscore_avg,
            ),
            index=["estimated_policy_value"],
        )
        policy_value_interval_df = DataFrame(
            self.estimate_intervals(
                action_dist=action_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                estimated_pscore=estimated_pscore,
                estimated_pscore_avg=estimated_pscore_avg,
                alpha=alpha,
                n_bootstrap_samples=n_bootstrap_samples,
                random_state=random_state,
            )
        )
        policy_value_of_behavior_policy = self.bandit_feedback["reward"].mean()
        policy_value_df = policy_value_df.T
        if policy_value_of_behavior_policy <= 0:
            logger.warning(
                f"Policy value of the behavior policy is {policy_value_of_behavior_policy} (<=0); relative estimated policy value is set to np.nan"
            )
            policy_value_df["relative_estimated_policy_value"] = np.nan
        else:
            policy_value_df["relative_estimated_policy_value"] = (
                policy_value_df.estimated_policy_value / policy_value_of_behavior_policy
            )
        return policy_value_df, policy_value_interval_df.T

    def visualize_off_policy_estimates(
        self,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: Optional[
            Union[np.ndarray, Dict[str, np.ndarray]]
        ] = None,
        estimated_pscore: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        estimated_pscore_avg: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        alpha: float = 0.05,
        is_relative: bool = False,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        fig_dir: Optional[Path] = None,
        fig_name: str = "estimated_policy_value.png",
    ) -> None:
        """Visualize the estimated policy values.

        Parameters
        ----------
        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list) or Dict[str, array-like], default=None
            Estimated expected rewards given context, action, and position, i.e., :math:`\\hat{q}(x_i,a_i)`.
            When an array-like is given, all OPE estimators use it.
            When a dict with an estimator's name as its key is given, the corresponding value is used for the estimator.
            If None, model-dependent estimators such as DM and DR cannot be used.

        estimated_pscore: array-like, shape (n_rounds,), default=None
            Estimated behavior policy (propensity scores), i.e., :math:`\\hat{\\pi}_b(a_i|x_i)`.
            When an array-like is given, all OPE estimators use it.
            When a dict with an estimator's name as its key is given, the corresponding value is used for the estimator.

        estimated_pscore_avg: array-like, shape (n_rounds,), default=None
            Estimated average behavior policy (propensity scores), i.e., :math:`\\hat{\\pi}_{avg}(a_i|x_i)`.
            When an array-like is given, all OPE estimators use it.
            When a dict with an estimator's name as its key is given, the corresponding value is used for the estimator.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=100
            Number of resampling performed in bootstrap sampling.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        is_relative: bool, default=False,
            If True, the method visualizes the estimated policy values of evaluation policy
            relative to the ground-truth policy value of behavior policy.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If None, the figure will not be saved.

        fig_name: str, default="estimated_policy_value.png"
            Name of the bar figure.

        """
        if fig_dir is not None:
            assert isinstance(fig_dir, Path), "`fig_dir` must be a Path"
        if fig_name is not None:
            assert isinstance(fig_name, str), "`fig_dir` must be a string"

        estimated_round_rewards_dict = dict()
        estimator_inputs = self._create_estimator_inputs(
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            estimated_pscore=estimated_pscore,
            estimated_pscore_avg=estimated_pscore_avg,
        )
        for estimator_name, estimator in self.ope_estimators_.items():
            estimated_round_rewards_dict[
                estimator_name
            ] = estimator._estimate_round_rewards(**estimator_inputs[estimator_name])
        estimated_round_rewards_df = DataFrame(estimated_round_rewards_dict)
        estimated_round_rewards_df.rename(
            columns={key: key.upper() for key in estimated_round_rewards_dict.keys()},
            inplace=True,
        )
        if is_relative:
            estimated_round_rewards_df /= self.bandit_feedback["reward"].mean()

        plt.style.use("ggplot")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(
            data=estimated_round_rewards_df,
            ax=ax,
            ci=100 * (1 - alpha),
            n_boot=n_bootstrap_samples,
            seed=random_state,
        )
        plt.xlabel("OPE Estimators", fontsize=25)
        plt.ylabel(
            f"Estimated Policy Value (± {np.int32(100*(1 - alpha))}% CI)", fontsize=20
        )
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=25 - 2 * len(self.ope_estimators))

        if fig_dir:
            fig.savefig(str(fig_dir / fig_name))

    def evaluate_performance_of_estimators(
        self,
        ground_truth_policy_value: float,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: Optional[
            Union[np.ndarray, Dict[str, np.ndarray]]
        ] = None,
        estimated_pscore: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        estimated_pscore_avg: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        metric: str = "se",
    ) -> Dict[str, float]:
        """Evaluate the accuracy of OPE estimators.

        Note
        ------
        Evaluate the estimation performance of OPE estimators with relative estimation error (relative-EE) or squared error (SE):

        .. math ::

            \\text{Relative-EE} (\\hat{V}; \\mathcal{D}) = \\left|  \\frac{\\hat{V}(\\pi; \\mathcal{D}) - V(\\pi)}{V(\\pi)} \\right|,

        .. math ::

            \\text{SE} (\\hat{V}; \\mathcal{D}) = \\left(\\hat{V}(\\pi; \\mathcal{D}) - V(\\pi) \\right)^2,

        where :math:`V({\\pi})` is the ground-truth policy value of the evalation policy :math:`\\pi_e` (often estimated using on-policy estimation).
        :math:`\\hat{V}(\\pi; \\mathcal{D})` is the policy value estimated by an OPE estimator :math:`\\hat{V}` and logged bandit feedback :math:`\\mathcal{D}`.

        Parameters
        ----------
        ground_truth policy value: float
            Ground_truth policy value of evaluation policy, i.e., :math:`V(\\pi_e)`.
            With Open Bandit Dataset, we use an on-policy estimate of the policy value as its ground-truth.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list) or Dict[str, array-like], default=None
            Estimated expected rewards given context, action, and position, i.e., :math:`\\hat{q}(x_i,a_i)`.
            When an array-like is given, all OPE estimators use it.
            When a dict with an estimator's name as its key is given, the corresponding value is used for the estimator.
            If None, model-dependent estimators such as DM and DR cannot be used.

        estimated_pscore: array-like, shape (n_rounds,), default=None
            Estimated behavior policy (propensity scores), i.e., :math:`\\hat{\\pi}_b(a_i|x_i)`.
            When an array-like is given, all OPE estimators use it.
            When a dict with an estimator's name as its key is given, the corresponding value is used for the estimator.

        estimated_pscore_avg: array-like, shape (n_rounds,), default=None
            Estimated average behavior policy (propensity scores), i.e., :math:`\\hat{\\pi}_{avg}(a_i|x_i)`.
            When an array-like is given, all OPE estimators use it.
            When a dict with an estimator's name as its key is given, the corresponding value is used for the estimator.

        metric: str, default="se"
            Evaluation metric used to evaluate and compare the estimation performance of OPE estimators.
            Must be either "relative-ee" or "se".

        Returns
        ----------
        eval_metric_ope_dict: Dict[str, float]
            Dictionary containing the value of evaluation metric for the estimation performance of OPE estimators.

        """
        check_scalar(
            ground_truth_policy_value,
            "ground_truth_policy_value",
            float,
        )
        if metric not in ["relative-ee", "se"]:
            raise ValueError(
                f"`metric` must be either 'relative-ee' or 'se', but {metric} is given"
            )
        if metric == "relative-ee" and ground_truth_policy_value == 0.0:
            raise ValueError(
                "`ground_truth_policy_value` must be non-zero when metric is relative-ee"
            )

        eval_metric_ope_dict = dict()
        estimator_inputs = self._create_estimator_inputs(
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
            estimated_pscore=estimated_pscore,
            estimated_pscore_avg=estimated_pscore_avg,
        )
        for estimator_name, estimator in self.ope_estimators_.items():
            estimated_policy_value = estimator.estimate_policy_value(
                **estimator_inputs[estimator_name]
            )
            if metric == "relative-ee":
                relative_ee_ = estimated_policy_value - ground_truth_policy_value
                relative_ee_ /= ground_truth_policy_value
                eval_metric_ope_dict[estimator_name] = np.abs(relative_ee_)
            elif metric == "se":
                se_ = (estimated_policy_value - ground_truth_policy_value) ** 2
                eval_metric_ope_dict[estimator_name] = se_
        return eval_metric_ope_dict

    def summarize_estimators_comparison(
        self,
        ground_truth_policy_value: float,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: Optional[
            Union[np.ndarray, Dict[str, np.ndarray]]
        ] = None,
        estimated_pscore: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        estimated_pscore_avg: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        metric: str = "se",
    ) -> DataFrame:
        """Summarize the performance comparison among OPE estimators.

        Parameters
        ----------
        ground_truth policy value: float
            Ground_truth policy value of evaluation policy, i.e., :math:`V(\\pi_e)`.
            With Open Bandit Dataset, we use an on-policy estimate of the policy value as ground-truth.

        action_dist: array-like, shape (n_rounds, n_actions, len_list)
            Action choice probabilities of the evaluation policy (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list), default=None
            Estimated expected rewards given context, action, and position, i.e., :math:`\\hat{q}(x_i,a_i)`.
            If None, model-dependent estimators such as DM and DR cannot be used.

        estimated_pscore: array-like, shape (n_rounds,), default=None
            Estimated behavior policy (propensity scores), i.e., :math:`\\hat{\\pi}_b(a_i|x_i)`.
            When an array-like is given, all OPE estimators use it.
            When a dict with an estimator's name as its key is given, the corresponding value is used for the estimator.

        estimated_pscore_avg: array-like, shape (n_rounds,), default=None
            Estimated average behavior policy (propensity scores), i.e., :math:`\\hat{\\pi}_{avg}(a_i|x_i)`.
            When an array-like is given, all OPE estimators use it.
            When a dict with an estimator's name as its key is given, the corresponding value is used for the estimator.

        metric: str, default="se"
            Evaluation metric used to evaluate and compare the estimation performance of OPE estimators.
            Must be either "relative-ee" or "se".

        Returns
        ----------
        eval_metric_ope_df: DataFrame
            Results of performance comparison among OPE estimators.

        """
        eval_metric_ope_df = DataFrame(
            self.evaluate_performance_of_estimators(
                ground_truth_policy_value=ground_truth_policy_value,
                action_dist=action_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                estimated_pscore=estimated_pscore,
                estimated_pscore_avg=estimated_pscore_avg,
                metric=metric,
            ),
            index=[metric],
        )
        return eval_metric_ope_df.T

    def visualize_off_policy_estimates_of_multiple_policies(
        self,
        policy_name_list: List[str],
        action_dist_list: List[np.ndarray],
        estimated_rewards_by_reg_model: Optional[
            Union[np.ndarray, Dict[str, np.ndarray]]
        ] = None,
        estimated_pscore: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        estimated_pscore_avg: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        alpha: float = 0.05,
        is_relative: bool = False,
        n_bootstrap_samples: int = 100,
        random_state: Optional[int] = None,
        fig_dir: Optional[Path] = None,
        fig_name: str = "estimated_policy_value.png",
    ) -> None:
        """Visualize the estimated policy values.

        Parameters
        ----------
        policy_name_list: List[str]
            List of the names of evaluation policies.

        action_dist_list: List[array-like, shape (n_rounds, n_actions, len_list)]
            List of action choice probabilities of the evaluation policies (can be deterministic), i.e., :math:`\\pi_e(a_i|x_i)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds, n_actions, len_list) or Dict[str, array-like], default=None
            Estimated expected rewards given context, action, and position, i.e., :math:`\\hat{q}(x_i,a_i)`.
            When an array-like is given, all OPE estimators use it.
            When a dict is given, if the dict has the name of an estimator as a key, the corresponding value is used.
            If None, model-dependent estimators such as DM and DR cannot be used.

        estimated_pscore: array-like, shape (n_rounds,), default=None
            Estimated behavior policy (propensity scores), i.e., :math:`\\hat{\\pi}_b(a_i|x_i)`.
            When an array-like is given, all OPE estimators use it.
            When a dict with an estimator's name as its key is given, the corresponding value is used for the estimator.

        estimated_pscore_avg: array-like, shape (n_rounds,), default=None
            Estimated average behavior policy (propensity scores), i.e., :math:`\\hat{\\pi}_{avg}(a_i|x_i)`.
            When an array-like is given, all OPE estimators use it.
            When a dict with an estimator's name as its key is given, the corresponding value is used for the estimator.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=100
            Number of resampling performed in bootstrap sampling.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        is_relative: bool, default=False,
            If True, the method visualizes the estimated policy values of evaluation policy
            relative to the ground-truth policy value of behavior policy.

        fig_dir: Path, default=None
            Path to store the bar figure.
            If None, the figure will not be saved.

        fig_name: str, default="estimated_policy_value.png"
            Name of the bar figure.

        """
        if len(policy_name_list) != len(action_dist_list):
            raise ValueError(
                "the length of `policy_name_list` must be the same as `action_dist_list`"
            )
        if fig_dir is not None:
            assert isinstance(fig_dir, Path), "`fig_dir` must be a Path"
        if fig_name is not None:
            assert isinstance(fig_name, str), "`fig_dir` must be a string"

        estimated_round_rewards_dict = {
            estimator_name: {} for estimator_name in self.ope_estimators_
        }

        for policy_name, action_dist in zip(policy_name_list, action_dist_list):
            estimator_inputs = self._create_estimator_inputs(
                action_dist=action_dist,
                estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
                estimated_pscore=estimated_pscore,
                estimated_pscore_avg=estimated_pscore_avg,
            )
            for estimator_name, estimator in self.ope_estimators_.items():
                estimated_round_rewards_dict[estimator_name][
                    policy_name
                ] = estimator._estimate_round_rewards(
                    **estimator_inputs[estimator_name]
                )

        plt.style.use("ggplot")
        fig = plt.figure(figsize=(8, 6.2 * len(self.ope_estimators_)))

        for i, estimator_name in enumerate(self.ope_estimators_):
            estimated_round_rewards_df = DataFrame(
                estimated_round_rewards_dict[estimator_name]
            )
            if is_relative:
                estimated_round_rewards_df /= self.bandit_feedback["reward"].mean()

            ax = fig.add_subplot(len(action_dist_list), 1, i + 1)
            sns.barplot(
                data=estimated_round_rewards_df,
                ax=ax,
                ci=100 * (1 - alpha),
                n_boot=n_bootstrap_samples,
                seed=random_state,
            )
            ax.set_title(estimator_name.upper(), fontsize=20)
            ax.set_ylabel(
                f"Estimated Policy Value (± {np.int32(100*(1 - alpha))}% CI)",
                fontsize=20,
            )
            plt.yticks(fontsize=15)
            plt.xticks(fontsize=25 - 2 * len(policy_name_list))

        if fig_dir:
            fig.savefig(str(fig_dir / fig_name))
