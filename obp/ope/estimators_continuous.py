# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Off-Policy Estimators for Continuous Actions."""
from abc import ABCMeta
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict
from typing import Optional

import numpy as np
from sklearn.utils import check_scalar

from ..utils import check_array
from ..utils import check_continuous_ope_inputs
from ..utils import estimate_confidence_interval_by_bootstrap


# kernel functions, reference: https://en.wikipedia.org/wiki/Kernel_(statistics)


def triangular_kernel(u: np.ndarray) -> np.ndarray:
    """Calculate triangular kernel function."""
    clipped_u = np.clip(u, -1.0, 1.0)
    return 1 - np.abs(clipped_u)


def gaussian_kernel(u: np.ndarray) -> np.ndarray:
    """Calculate gaussian kernel function."""
    return np.exp(-(u**2) / 2) / np.sqrt(2 * np.pi)


def epanechnikov_kernel(u: np.ndarray) -> np.ndarray:
    """Calculate epanechnikov kernel function."""
    clipped_u = np.clip(u, -1.0, 1.0)
    return 0.75 * (1 - clipped_u**2)


def cosine_kernel(u: np.ndarray) -> np.ndarray:
    """Calculate cosine kernel function."""
    clipped_u = np.clip(u, -1.0, 1.0)
    return (np.pi / 4) * np.cos(clipped_u * np.pi / 2)


kernel_functions = dict(
    gaussian=gaussian_kernel,
    epanechnikov=epanechnikov_kernel,
    triangular=triangular_kernel,
    cosine=cosine_kernel,
)


@dataclass
class BaseContinuousOffPolicyEstimator(metaclass=ABCMeta):
    """Base class for OPE estimators for continuous actions."""

    @abstractmethod
    def _estimate_round_rewards(self) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards."""
        raise NotImplementedError

    @abstractmethod
    def estimate_policy_value(self) -> float:
        """Estimate policy value of evaluation policy."""
        raise NotImplementedError

    @abstractmethod
    def estimate_interval(self) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap."""
        raise NotImplementedError


@dataclass
class KernelizedInverseProbabilityWeighting(BaseContinuousOffPolicyEstimator):
    """Kernelized Inverse Probability Weighting.

    Note
    -------
    Kernel IPW estimates the policy value of a given (deterministic) evaluation policy :math:`\\pi_e` as

    .. math::

        \\hat{V}_{\\mathrm{Kernel-IPW}} (\\pi_e; \\mathcal{D})
        := \\mathbb{E}_{n} \\left[ \\frac{1}{h} K \\left( \\frac{\\pi_e(x_i) - a_i}{h} \\right) \\frac{r_i}{q_i} \\right],

    where :math:`\\mathcal{D}=\\{(x_i,a_i,r_i)\\}_{i=1}^{n}` is logged bandit data with :math:`n` observations collected by behavior policy.
    Note that each action :math:`a_t` in the logged bandit data is a continuous variable.
    :math:`q_i` is the generalized propensity score, which is defined as the conditional probability density of the behavior policy.
    :math:`K(\cdot)` is a kernel function such as the gaussian kernel, and :math:`h` is a bandwidth hyperparameter.
    :math:`\\pi_e (x)` is a deterministic evaluation policy that maps :math:`x` to a continuous action value.
    :math:`\\mathbb{E}_{n}[\\cdot]` is the empirical average over :math:`n` observations in :math:`\\mathcal{D}`.

    Parameters
    ------------
    kernel: str
        Choice of kernel function.
        Must be one of "gaussian", "epanechnikov", "triangular", or "cosine".

    bandwidth: float
        Bandwidth hyperparameter.
        A larger value increases bias instead of reducing variance.
        A smaller value increases variance instead of reducing bias.

    estimator_name: str, default='kernelized_ipw'.
        Name of the estimator.

    References
    ------------
    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments", 2018.

    """

    kernel: str
    bandwidth: float
    estimator_name: str = "kernelized_ipw"

    def __post_init__(self) -> None:
        if self.kernel not in ["gaussian", "epanechnikov", "triangular", "cosine"]:
            raise ValueError(
                f"kernel must be one of 'gaussian', 'epanechnikov', 'triangular', or 'cosine' but {self.kernel} is given"
            )
        check_scalar(
            self.bandwidth, name="bandwidth", target_type=(int, float), min_val=0
        )

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action_by_behavior_policy: np.ndarray,
        pscore: np.ndarray,
        action_by_evaluation_policy: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action_by_behavior_policy: array-like, shape (n_rounds,)
            Continuous action values sampled by behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        pscore: array-like, shape (n_rounds,)
            Probability densities of the continuous action values sampled by behavior policy
            (generalized propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.

        action_by_evaluation_policy: array-like, shape (n_rounds,)
            Continuous action values given by evaluation policy (can be deterministic), i.e., :math:`\\pi_e(x_t)`.

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Estimated rewards for each observation.

        """
        kernel_func = kernel_functions[self.kernel]
        u = action_by_evaluation_policy - action_by_behavior_policy
        u /= self.bandwidth
        estimated_rewards = kernel_func(u) * reward / pscore
        estimated_rewards /= self.bandwidth
        return estimated_rewards

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action_by_behavior_policy: np.ndarray,
        pscore: np.ndarray,
        action_by_evaluation_policy: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Estimate policy value of evaluation policy.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action_by_behavior_policy: array-like, shape (n_rounds,)
            Continuous action values sampled by behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        pscore: array-like, shape (n_rounds,)
            Probability densities of the continuous action values sampled by behavior policy
            (generalized propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.

        action_by_evaluation_policy: array-like, shape (n_rounds,)
            Continuous action values given by evaluation policy (can be deterministic), i.e., :math:`\\pi_e(x_t)`.

        Returns
        ----------
        V_hat: float
            Estimated policy value of evaluation policy.

        """
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(
            array=action_by_behavior_policy,
            name="action_by_behavior_policy",
            expected_dim=1,
        )
        check_array(array=pscore, name="pscore", expected_dim=1)
        check_continuous_ope_inputs(
            reward=reward,
            action_by_behavior_policy=action_by_behavior_policy,
            pscore=pscore,
            action_by_evaluation_policy=action_by_evaluation_policy,
        )

        return self._estimate_round_rewards(
            reward=reward,
            action_by_behavior_policy=action_by_behavior_policy,
            pscore=pscore,
            action_by_evaluation_policy=action_by_evaluation_policy,
        ).mean()

    def estimate_interval(
        self,
        reward: np.ndarray,
        action_by_behavior_policy: np.ndarray,
        pscore: np.ndarray,
        action_by_evaluation_policy: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action_by_behavior_policy: array-like, shape (n_rounds,)
            Continuous action values sampled by behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        pscore: array-like, shape (n_rounds,)
            Probability densities of the continuous action values sampled by behavior policy
            (generalized propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.

        action_by_evaluation_policy: array-like, shape (n_rounds,)
            Continuous action values given by evaluation policy (can be deterministic), i.e., :math:`\\pi_e(x_t)`.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in bootstrap sampling.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(
            array=action_by_behavior_policy,
            name="action_by_behavior_policy",
            expected_dim=1,
        )
        check_array(array=pscore, name="pscore", expected_dim=1)
        check_continuous_ope_inputs(
            reward=reward,
            action_by_behavior_policy=action_by_behavior_policy,
            pscore=pscore,
            action_by_evaluation_policy=action_by_evaluation_policy,
        )

        estimated_round_rewards = self._estimate_round_rewards(
            reward=reward,
            action_by_behavior_policy=action_by_behavior_policy,
            pscore=pscore,
            action_by_evaluation_policy=action_by_evaluation_policy,
        )

        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class KernelizedSelfNormalizedInverseProbabilityWeighting(
    BaseContinuousOffPolicyEstimator
):
    """Kernelized Self-Normalized Inverse Probability Weighting.

    Note
    -------
    Kernel SNIPW estimates the policy value of a given (deterministic) evaluation policy :math:`\\pi_e` as

    .. math::

        \\hat{V}_{\\mathrm{Kernel-SNIPW}} (\\pi_e; \\mathcal{D})
        := \\frac{\\mathbb{E}_{n} \\left[ K \\left( \\frac{\\pi_e(x_i) - a_i}{h} \\right) \\frac{r_i}{q_i} \\right]}{\\mathbb{E}_{n} \\left[ K \\left( \\frac{\\pi_e(x_i) - a_i}{h} \\right) \\frac{r_i}{q_i}},

    where :math:`\\mathcal{D}=\\{(x_i,a_i,r_i)\\}_{i=1}^{n}` is logged bandit data with :math:`n` observations collected by behavior policy.
    Note that each action :math:`a_t` in the logged bandit data is a continuous variable.
    :math:`q_i` is the generalized propensity score, which is defined as the conditional probability density of the behavior policy.
    :math:`K(\cdot)` is a kernel function such as the gaussian kernel, and :math:`h` is a bandwidth hyperparameter.
    :math:`\\pi_e (x)` is a deterministic evaluation policy that maps :math:`x` to a continuous action value.
    :math:`\\mathbb{E}_{n}[\\cdot]` is the empirical average over :math:`n` observations in :math:`\\mathcal{D}`.

    Parameters
    ------------
    kernel: str
        Choice of kernel function.
        Must be one of "gaussian", "epanechnikov", "triangular", or "cosine".

     bandwidth: float
        Bandwidth hyperparameter.
        A larger value increases bias instead of reducing variance.
        A smaller value increases variance instead of reducing bias.

    estimator_name: str, default='kernelized_snipw'.
        Name of the estimator.

    References
    ------------
    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments", 2018.

    """

    kernel: str
    bandwidth: float
    estimator_name: str = "kernelized_snipw"

    def __post_init__(self) -> None:
        if self.kernel not in ["gaussian", "epanechnikov", "triangular", "cosine"]:
            raise ValueError(
                f"kernel must be one of 'gaussian', 'epanechnikov', 'triangular', or 'cosine' but {self.kernel} is given"
            )
        check_scalar(
            self.bandwidth, name="bandwidth", target_type=(int, float), min_val=0
        )

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action_by_behavior_policy: np.ndarray,
        pscore: np.ndarray,
        action_by_evaluation_policy: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action_by_behavior_policy: array-like, shape (n_rounds,)
            Continuous action values sampled by behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        pscore: array-like, shape (n_rounds,)
            Probability densities of the continuous action values sampled by behavior policy
            (generalized propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.

        action_by_evaluation_policy: array-like, shape (n_rounds,)
            Continuous action values given by evaluation policy (can be deterministic), i.e., :math:`\\pi_e(x_t)`.

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Estimated rewards for each observation.

        """
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(
            array=action_by_behavior_policy,
            name="action_by_behavior_policy",
            expected_dim=1,
        )
        check_array(array=pscore, name="pscore", expected_dim=1)

        kernel_func = kernel_functions[self.kernel]
        u = action_by_evaluation_policy - action_by_behavior_policy
        u /= self.bandwidth
        estimated_rewards = kernel_func(u) * reward / pscore
        estimated_rewards /= (kernel_func(u) / pscore).mean()
        return estimated_rewards

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action_by_behavior_policy: np.ndarray,
        pscore: np.ndarray,
        action_by_evaluation_policy: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Estimate policy value of evaluation policy.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action_by_behavior_policy: array-like, shape (n_rounds,)
            Continuous action values sampled by behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        pscore: array-like, shape (n_rounds,)
            Probability densities of the continuous action values sampled by behavior policy
            (generalized propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.

        action_by_evaluation_policy: array-like, shape (n_rounds,)
            Continuous action values given by evaluation policy (can be deterministic), i.e., :math:`\\pi_e(x_t)`.

        Returns
        ----------
        V_hat: float
            Estimated policy value of evaluation policy.

        """
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(
            array=action_by_behavior_policy,
            name="action_by_behavior_policy",
            expected_dim=1,
        )
        check_array(array=pscore, name="pscore", expected_dim=1)
        check_continuous_ope_inputs(
            reward=reward,
            action_by_behavior_policy=action_by_behavior_policy,
            pscore=pscore,
            action_by_evaluation_policy=action_by_evaluation_policy,
        )

        return self._estimate_round_rewards(
            reward=reward,
            action_by_behavior_policy=action_by_behavior_policy,
            pscore=pscore,
            action_by_evaluation_policy=action_by_evaluation_policy,
        ).mean()

    def estimate_interval(
        self,
        reward: np.ndarray,
        action_by_behavior_policy: np.ndarray,
        pscore: np.ndarray,
        action_by_evaluation_policy: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action_by_behavior_policy: array-like, shape (n_rounds,)
            Continuous action values sampled by behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        pscore: array-like, shape (n_rounds,)
            Probability densities of the continuous action values sampled by behavior policy
            (generalized propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.

        action_by_evaluation_policy: array-like, shape (n_rounds,)
            Continuous action values given by evaluation policy (can be deterministic), i.e., :math:`\\pi_e(x_t)`.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in bootstrap sampling.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(
            array=action_by_behavior_policy,
            name="action_by_behavior_policy",
            expected_dim=1,
        )
        check_array(array=pscore, name="pscore", expected_dim=1)
        check_continuous_ope_inputs(
            reward=reward,
            action_by_behavior_policy=action_by_behavior_policy,
            pscore=pscore,
            action_by_evaluation_policy=action_by_evaluation_policy,
        )

        estimated_round_rewards = self._estimate_round_rewards(
            reward=reward,
            action_by_behavior_policy=action_by_behavior_policy,
            pscore=pscore,
            action_by_evaluation_policy=action_by_evaluation_policy,
        )

        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )


@dataclass
class KernelizedDoublyRobust(BaseContinuousOffPolicyEstimator):
    """Kernelized Doubly Robust Estimator.

    Note
    -------
    Kernel DR estimates the policy value of a given (deterministic) evaluation policy :math:`\\pi_e` as

    .. math::

        \\hat{V}_{\\mathrm{Kernel-DR}} (\\pi_e; \\mathcal{D})
        := \\mathbb{E}_{n} \\left[ \\frac{1}{h} K \\left( \\frac{\\pi_e(x_i) - a_i}{h} \\right) \\frac{(r_i - \\hat{q}(x_i, \\pi_e(x_t)))}{q_t} + \\hat{q}(x_i, \\pi_e(x_t)) \\right],

    where :math:`\\mathcal{D}=\\{(x_i,a_i,r_i)\\}_{i=1}^{n}` is logged bandit data with :math:`n` observations collected by behavior policy.
    Note that each action :math:`a_t` in the logged bandit data is a continuous variable.
    :math:`q_i` is the generalized propensity score, which is defined as the conditional probability density of the behavior policy.
    :math:`K(\cdot)` is a kernel function such as the gaussian kernel, and :math:`h` is a bandwidth hyperparameter.
    :math:`\\pi_e (x)` is a deterministic evaluation policy that maps :math:`x` to a continuous action value.
    :math:`\\hat{q} (x,a)` is the estimated expected reward given :math:`x` and :math:`a`.
    :math:`\\mathbb{E}_{n}[\\cdot]` is the empirical average over :math:`n` observations in :math:`\\mathcal{D}`.

    Parameters
    ------------
    kernel: str
        Choice of kernel function.
        Must be one of "gaussian", "epanechnikov", "triangular", or "cosine".

     bandwidth: float
        Bandwidth hyperparameter.
        A larger value increases bias instead of reducing variance.
        A smaller value increases variance instead of reducing bias.

    estimator_name: str, default='kernelized_dr'.
        Name of the estimator.

    References
    ------------
    Nathan Kallus and Angela Zhou.
    "Policy Evaluation and Optimization with Continuous Treatments", 2018.

    """

    kernel: str
    bandwidth: float
    estimator_name: str = "kernelized_dr"

    def __post_init__(self) -> None:
        if self.kernel not in ["gaussian", "epanechnikov", "triangular", "cosine"]:
            raise ValueError(
                f"kernel must be one of 'gaussian', 'epanechnikov', 'triangular', or 'cosine' but {self.kernel} is given"
            )
        check_scalar(
            self.bandwidth, name="bandwidth", target_type=(int, float), min_val=0
        )

    def _estimate_round_rewards(
        self,
        reward: np.ndarray,
        action_by_behavior_policy: np.ndarray,
        pscore: np.ndarray,
        action_by_evaluation_policy: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Estimate round-wise (or sample-wise) rewards.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action_by_behavior_policy: array-like, shape (n_rounds,)
            Continuous action values sampled by behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        pscore: array-like, shape (n_rounds,)
            Probability densities of the continuous action values sampled by behavior policy
            (generalized propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.

        action_by_evaluation_policy: array-like, shape (n_rounds,)
            Continuous action values given by evaluation policy (can be deterministic), i.e., :math:`\\pi_e(x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds,)
            Expected rewards given context and action estimated by regression model, i.e., :math:`\\hat{q}(x_i,a_i)`.

        Returns
        ----------
        estimated_rewards: array-like, shape (n_rounds,)
            Estimated rewards for each observation.

        """
        kernel_func = kernel_functions[self.kernel]
        u = action_by_evaluation_policy - action_by_behavior_policy
        u /= self.bandwidth
        estimated_rewards = (
            kernel_func(u) * (reward - estimated_rewards_by_reg_model) / pscore
        )
        estimated_rewards /= self.bandwidth
        estimated_rewards += estimated_rewards_by_reg_model
        return estimated_rewards

    def estimate_policy_value(
        self,
        reward: np.ndarray,
        action_by_behavior_policy: np.ndarray,
        pscore: np.ndarray,
        action_by_evaluation_policy: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Estimate policy value of evaluation policy.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action_by_behavior_policy: array-like, shape (n_rounds,)
            Continuous action values sampled by behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        pscore: array-like, shape (n_rounds,)
            Probability densities of the continuous action values sampled by behavior policy
            (generalized propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.

        action_by_evaluation_policy: array-like, shape (n_rounds,)
            Continuous action values given by evaluation policy (can be deterministic), i.e., :math:`\\pi_e(x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds,)
            Expected rewards given context and action estimated by regression model, i.e., :math:`\\hat{q}(x_i,a_i)`.

        Returns
        ----------
        V_hat: float
            Estimated policy value of evaluation policy.

        """
        check_array(
            array=estimated_rewards_by_reg_model,
            name="estimated_rewards_by_reg_model",
            expected_dim=1,
        )
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(
            array=action_by_behavior_policy,
            name="action_by_behavior_policy",
            expected_dim=1,
        )
        check_array(array=pscore, name="pscore", expected_dim=1)
        check_continuous_ope_inputs(
            reward=reward,
            action_by_behavior_policy=action_by_behavior_policy,
            pscore=pscore,
            action_by_evaluation_policy=action_by_evaluation_policy,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )

        return self._estimate_round_rewards(
            reward=reward,
            action_by_behavior_policy=action_by_behavior_policy,
            pscore=pscore,
            action_by_evaluation_policy=action_by_evaluation_policy,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        ).mean()

    def estimate_interval(
        self,
        reward: np.ndarray,
        action_by_behavior_policy: np.ndarray,
        pscore: np.ndarray,
        action_by_evaluation_policy: np.ndarray,
        estimated_rewards_by_reg_model: np.ndarray,
        alpha: float = 0.05,
        n_bootstrap_samples: int = 10000,
        random_state: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """Estimate the confidence interval of the policy value using bootstrap.

        Parameters
        ----------
        reward: array-like, shape (n_rounds,)
            Rewards observed for each data in logged bandit data, i.e., :math:`r_i`.

        action_by_behavior_policy: array-like, shape (n_rounds,)
            Continuous action values sampled by behavior policy for each data in logged bandit data, i.e., :math:`a_i`.

        pscore: array-like, shape (n_rounds,)
            Probability densities of the continuous action values sampled by behavior policy
            (generalized propensity scores), i.e., :math:`\\pi_b(a_i|x_i)`.

        action_by_evaluation_policy: array-like, shape (n_rounds,)
            Continuous action values given by evaluation policy (can be deterministic), i.e., :math:`\\pi_e(x_t)`.

        estimated_rewards_by_reg_model: array-like, shape (n_rounds,)
            Expected rewards given context and action estimated by regression model, i.e., :math:`\\hat{q}(x_i,a_i)`.

        alpha: float, default=0.05
            Significance level.

        n_bootstrap_samples: int, default=10000
            Number of resampling performed in bootstrap sampling.

        random_state: int, default=None
            Controls the random seed in bootstrap sampling.

        Returns
        ----------
        estimated_confidence_interval: Dict[str, float]
            Dictionary storing the estimated mean and upper-lower confidence bounds.

        """
        check_array(
            array=estimated_rewards_by_reg_model,
            name="estimated_rewards_by_reg_model",
            expected_dim=1,
        )
        check_array(array=reward, name="reward", expected_dim=1)
        check_array(
            array=action_by_behavior_policy,
            name="action_by_behavior_policy",
            expected_dim=1,
        )
        check_array(array=pscore, name="pscore", expected_dim=1)
        check_continuous_ope_inputs(
            reward=reward,
            action_by_behavior_policy=action_by_behavior_policy,
            pscore=pscore,
            action_by_evaluation_policy=action_by_evaluation_policy,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )

        estimated_round_rewards = self._estimate_round_rewards(
            reward=reward,
            action_by_behavior_policy=action_by_behavior_policy,
            pscore=pscore,
            action_by_evaluation_policy=action_by_evaluation_policy,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )

        return estimate_confidence_interval_by_bootstrap(
            samples=estimated_round_rewards,
            alpha=alpha,
            n_bootstrap_samples=n_bootstrap_samples,
            random_state=random_state,
        )
