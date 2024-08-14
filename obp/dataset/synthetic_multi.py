# Copyright (c) Yuta Saito, Yusuke Narita, and ZOZO Technologies, Inc. All rights reserved.
# Licensed under the Apache 2.0 License.

"""Class for Generating Synthetic Logged Bandit Data with Multiple Loggers."""
from dataclasses import dataclass
from typing import List
from typing import Union

import numpy as np
from scipy.stats import truncnorm
from sklearn.utils import check_random_state
from sklearn.utils import check_scalar

from ..types import BanditFeedback
from ..utils import check_array
from ..utils import sample_action_fast
from ..utils import softmax
from .reward_type import RewardType
from .synthetic import SyntheticBanditDataset


@dataclass
class SyntheticMultiLoggersBanditDataset(SyntheticBanditDataset):
    """Class for synthesizing bandit data with multiple logging/behavior policies.

    Note
    -----
    By calling the `obtain_batch_bandit_feedback` method several times,
    we can resample logged bandit data from the same data generating distribution.
    This can be used to estimate confidence intervals of the performances of OPE estimators.

    If None is given as `behavior_policy_function`, the behavior policy will be generated from the true expected reward function. See the description of the `beta` argument, which controls the behavior policy.

    Parameters
    -----------
    n_actions: int
        Number of actions.

    betas: List[Union[int, float]]
        A list of inverse temperature parameters, which controls the optimality and entropy of the behavior policies.
        A large value leads to a near-deterministic behavior policy,
        while a small value leads to a near-uniform behavior policy.
        A positive value leads to a near-optimal behavior policy,
        while a negative value leads to a sub-optimal behavior policy.
        Note that in the implementation, None is set as default, but a list must be set to use this class
        to generate synthetic data with multiple loggers.

    rhos: List[Union[int, float]]
        A list of propotions of strata sizes to generate logged bandit data with multiple loggers.
        For example, when `[0.1, 0.4, 0.5]` is given as `rhos`, there will be three strata, each stratum is generated by
        different logging/behavior policies and has data size defined by the corresponding propotion
        in the list (sum up to 1).
        Note that in the implementation, None is set as default, but a list must be set to use this class
        to generate synthetic data with multiple loggers.
        Moreover, `len(betas) == len(rhos)` should be met.

    dim_context: int, default=1
        Number of dimensions of context vectors.

    reward_type: str, default='binary'
        Type of reward variable, which must be either 'binary' or 'continuous'.
        When 'binary', rewards are sampled from the Bernoulli distribution.
        When 'continuous', rewards are sampled from the truncated Normal distribution with `scale=1`.
        The mean parameter of the reward distribution is determined by the `reward_function` specified by the next argument.

    reward_function: Callable[[np.ndarray, np.ndarray], np.ndarray]], default=None
        Function defining the expected reward for each given action-context pair,
        i.e., :math:`q: \\mathcal{X} \\times \\mathcal{A} \\rightarrow \\mathbb{R}`.
        If None, context **independent** expected rewards will be
        sampled from the uniform distribution automatically.

    reward_std: float, default=1.0
        Standard deviation of the reward distribution.
        A larger value leads to a noisier reward distribution.
        This argument is valid only when `reward_type="continuous"`.

    action_context: np.ndarray, default=None
         Vector representation of (discrete) actions.
         If None, one-hot representation will be used.

    n_deficient_actions: int, default=0
        Number of deficient actions having zero probability of being selected in the logged bandit data.
        If there are some deficient actions, the full/common support assumption is very likely to be violated,
        leading to some bias for IPW-type estimators. See Sachdeva et al.(2020) for details.
        `n_deficient_actions` should be an integer smaller than `n_actions - 1` so that there exists at least one action
        that have a positive probability of being selected by the behavior policy.

    random_state: int, default=12345
        Controls the random seed in sampling synthetic bandit data.

    dataset_name: str, default='synthetic_bandit_dataset'
        Name of the dataset.

    Examples
    ----------

    .. code-block:: python

        >>> from obp.dataset import (
            SyntheticMultiLoggersBanditDataset,
            logistic_reward_function
        )

        # generate synthetic contextual bandit feedback with 10 actions.
        >>> dataset = SyntheticBanditDataset(
                n_actions=3,
                dim_context=5,
                reward_function=logistic_reward_function,
                betas=[-3, 0, 3],
                rhos=[0.2, 0.5, 0.3],
                random_state=12345
            )
        >>> bandit_feedback = dataset.obtain_batch_bandit_feedback(n_rounds=100000)
        >>> bandit_feedback
            {
                'n_rounds': 10000,
                'n_actions': 5,
                'n_strata': 3,
                'context': array([[-0.20470766,  0.47894334, -0.51943872, -0.5557303 ,  1.96578057],
                        [ 1.39340583,  0.09290788,  0.28174615,  0.76902257,  1.24643474],
                        [ 1.00718936, -1.29622111,  0.27499163,  0.22891288,  1.35291684],
                        ...,
                        [-1.27028221,  0.80914602, -0.45084222,  0.47179511,  1.89401115],
                        [-0.68890924,  0.08857502, -0.56359347, -0.41135069,  0.65157486],
                        [ 0.51204121,  0.65384817, -1.98849253, -2.14429131, -0.34186901]]),
                'action_context': array([[1, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 1]]),
                'action': array([4, 0, 0, ..., 0, 2, 3]),
                'position': None,
                'reward': array([1, 0, 0, ..., 1, 1, 0]),
                'expected_reward': array([[0.43166951, 0.46506373, 0.62424214, 0.71722648, 0.74364688],
                        [0.45615482, 0.67859997, 0.68074447, 0.74845979, 0.70857809],
                        [0.49314956, 0.54930546, 0.66069967, 0.71459597, 0.84417897],
                        ...,
                        [0.25147493, 0.49371688, 0.56138603, 0.73893432, 0.6977549 ],
                        [0.38629307, 0.39534047, 0.56645931, 0.62379871, 0.70810635],
                        [0.7394717 , 0.39578418, 0.82279914, 0.65471639, 0.73325977]]),
                        [0.19032729],
                        [0.1965272 ]]]),
                'pscore': array([0.11948302, 0.34389468, 0.3018088 , ..., 0.07281752, 0.20415909,
                        0.17634078]),
                'pscore_avg': array([0.21089116, 0.20053648, 0.19505875, ..., 0.20616291, 0.19524095,
                        0.19032729])
            }

    References
    ------------
    Miroslav Dudík, Dumitru Erhan, John Langford, and Lihong Li.
    "Doubly Robust Policy Evaluation and Optimization.", 2014.

    Noveen Sachdeva, Yi Su, and Thorsten Joachims.
    "Off-policy Bandits with Deficient Support.", 2020.

    Aman Agarwal, Soumya Basu, Tobias Schnabel, Thorsten Joachims.
    "Effective Evaluation using Logged Bandit Feedback from Multiple Loggers.", 2018.

    Nathan Kallus, Yuta Saito, and Masatoshi Uehara.
    "Optimal Off-Policy Evaluation from Multiple Logging Policies.", 2020.
    """

    betas: List[Union[int, float]] = None
    rhos: List[float] = None

    def __post_init__(self) -> None:
        """Initialize Class."""
        check_scalar(self.n_actions, "n_actions", int, min_val=2)
        check_scalar(self.dim_context, "dim_context", int, min_val=1)
        check_scalar(
            self.n_deficient_actions,
            "n_deficient_actions",
            int,
            min_val=0,
            max_val=self.n_actions - 1,
        )

        if self.random_state is None:
            raise ValueError("`random_state` must be given")
        self.random_ = check_random_state(self.random_state)

        if self.behavior_policy_function is not None:
            print(
                "something is given as `behavior_policy_function`, but it will not be used in this class."
            )

        if not isinstance(self.betas, list):
            raise TypeError(f"`betas` must be a list, but {type(self.betas)} is given.")
        else:
            for k, beta in enumerate(self.betas):
                check_scalar(beta, f"betas[{k}]", (int, float))
        if not isinstance(self.rhos, list):
            raise TypeError(f"`rhos` must be a list, but {type(self.betas)} is given.")
        else:
            for k, rho in enumerate(self.rhos):
                check_scalar(rho, f"rhos[{k}]", (int, float), min_val=0.0)
            self.rhos = np.array(self.rhos) / np.sum(self.rhos)
        if len(self.betas) != len(self.rhos):
            raise ValueError(
                "Expected `len(self.betas) == len(self.rhos)`, but Found it False."
            )

        if RewardType(self.reward_type) not in [
            RewardType.BINARY,
            RewardType.CONTINUOUS,
        ]:
            raise ValueError(
                f"`reward_type` must be either '{RewardType.BINARY.value}' or '{RewardType.CONTINUOUS.value}',"
                f"but {self.reward_type} is given.'"
            )
        check_scalar(self.reward_std, "reward_std", (int, float), min_val=0)
        if self.reward_function is None:
            self.expected_reward = self.sample_contextfree_expected_reward()
        if RewardType(self.reward_type) == RewardType.CONTINUOUS:
            self.reward_min = 0
            self.reward_max = 1e10

        # one-hot encoding characterizing actions.
        if self.action_context is None:
            self.action_context = np.eye(self.n_actions, dtype=int)
        else:
            check_array(
                array=self.action_context, name="action_context", expected_dim=2
            )
            if self.action_context.shape[0] != self.n_actions:
                raise ValueError(
                    "Expected `action_context.shape[0] == n_actions`, but found it False."
                )

        if self.random_state is None:
            raise ValueError("`random_state` must be given")
        self.random_ = check_random_state(self.random_state)

    @property
    def len_list(self) -> int:
        """Length of recommendation lists, slate size."""
        return 1

    @property
    def n_strata(self) -> int:
        """Number of strata, number of logging/behavior policies."""
        return len(self.betas)

    def obtain_batch_bandit_feedback(self, n_rounds: int) -> BanditFeedback:
        """Obtain batch logged bandit data.

        Parameters
        ----------
        n_rounds: int
            Total data size (= sum of size of each stratum) of the synthetic logged bandit data.

        Returns
        ---------
        bandit_feedback_multi: BanditFeedback
            Synthesized logged bandit data with multiple loggers.

        """
        check_scalar(n_rounds, "n_rounds", int, min_val=1)
        contexts = self.random_.normal(size=(n_rounds, self.dim_context))

        n_rounds_strata = [int(np.round(n_rounds * rho)) for rho in self.rhos]
        n_rounds_strata[-1] += n_rounds - np.sum(n_rounds_strata)
        stratum_idx = np.concatenate(
            [np.ones(n_k, dtype=int) * k for k, n_k in enumerate(n_rounds_strata)]
        )
        beta_ = np.concatenate(
            [np.ones(n_k) * self.betas[k] for k, n_k in enumerate(n_rounds_strata)]
        )[:, np.newaxis]

        # calc expected reward given context and action
        expected_reward_ = self.calc_expected_reward(contexts)
        if RewardType(self.reward_type) == RewardType.CONTINUOUS:
            # correct expected_reward_, as we use truncated normal distribution here
            mean = expected_reward_
            a = (self.reward_min - mean) / self.reward_std
            b = (self.reward_max - mean) / self.reward_std
            expected_reward_ = truncnorm.stats(
                a=a, b=b, loc=mean, scale=self.reward_std, moments="m"
            )

        # calculate the action choice probabilities of the behavior policy
        pi_b_logits = expected_reward_
        # create some deficient actions based on the value of `n_deficient_actions`
        if self.n_deficient_actions > 0:
            pi_b = np.zeros_like(pi_b_logits)
            n_supported_actions = self.n_actions - self.n_deficient_actions
            supported_actions = np.argsort(
                self.random_.gumbel(size=(n_rounds, self.n_actions)), axis=1
            )[:, ::-1][:, :n_supported_actions]
            supported_actions_idx = (
                np.tile(np.arange(n_rounds), (n_supported_actions, 1)).T,
                supported_actions,
            )
            pi_b[supported_actions_idx] = softmax(
                self.beta * pi_b_logits[supported_actions_idx]
            )
        else:
            pi_b = softmax(beta_ * pi_b_logits)
        # sample actions for each round based on the behavior policy
        actions = sample_action_fast(pi_b, random_state=self.random_state)

        pi_b_avg = np.zeros_like(pi_b_logits)
        for beta, rho in zip(self.betas, self.rhos):
            pi_b_avg += rho * softmax(beta * pi_b_logits)

        # sample rewards based on the context and action
        rewards = self.sample_reward_given_expected_reward(expected_reward_, actions)

        return dict(
            n_rounds=n_rounds,
            n_actions=self.n_actions,
            n_strata=self.n_strata,
            context=contexts,
            action_context=self.action_context,
            action=actions,
            position=None,  # position effect is not considered in synthetic data
            reward=rewards,
            expected_reward=expected_reward_,
            stratum_idx=stratum_idx,
            pi_b=pi_b[:, :, np.newaxis],
            pi_b_avg=pi_b_avg[:, :, np.newaxis],
            pscore=pi_b[np.arange(n_rounds), actions],
            pscore_avg=pi_b_avg[np.arange(n_rounds), actions],
        )
