
import dataclasses
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
)

from stable_baselines3.common import policies, utils, vec_env
import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common import utils
from imitation.algorithms import base as algo_base
from imitation.data import types
from imitation.util import logger as imit_logger
from imitation.util import util
from imitation.algorithms.bc import BehaviorCloningLossCalculator, BCTrainingMetrics, BatchIteratorWithEpochEndCallback, enumerate_batches, RolloutStatsComputer, BCLogger
from drl_utils.algorithms.gru_ppo.gru_actor_critic import GruActorCriticPolicy
import tqdm  
from drl_utils.algorithms.behavioral_clonings.bc import make_data_loader

class GruBCLogger(BCLogger):

    def __init__(self, logger: imit_logger.HierarchicalLogger):
        super().__init__(logger)

    def log_batch(
        self,
        batch_num: int,
        batch_size: int,
        num_samples_so_far: int,
        training_metrics: BCTrainingMetrics,
        rollout_stats: Mapping[str, float],
    ):
        self._logger.record("batch_size", batch_size)
        self._logger.record("bc/epoch", self._current_epoch)
        self._logger.record("bc/batch", batch_num)
        self._logger.record("bc/samples_so_far", num_samples_so_far)
        for k, v in training_metrics.__dict__.items():
            if not k == 'gru_states':
                self._logger.record(f"bc/{k}", float(v) if v is not None else None)

        for k, v in rollout_stats.items():
            if "return" in k and "monitor" not in k:
                self._logger.record("rollout/" + k, v)
        self._logger.dump(self._tensorboard_step)
        self._tensorboard_step += 1


@dataclasses.dataclass(frozen=True)
class GruBCTrainingMetrics:
    """Container for the different components of behavior cloning loss."""
    neglogp: th.Tensor
    entropy: Optional[th.Tensor]
    ent_loss: th.Tensor  # set to 0 if entropy is None
    prob_true_act: th.Tensor
    l2_norm: th.Tensor
    l2_loss: th.Tensor
    loss: th.Tensor
    gru_states:th.Tensor

def evaluate_actions(
    policy: GruActorCriticPolicy, 
    obs: th.Tensor, 
    actions: th.Tensor, 
    gru_states: th.Tensor, 
    episode_starts: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    features = policy.extract_features(obs)
    latent_pi, gru_states = policy._process_sequence(features, gru_states, episode_starts, policy.gru)
    latent_pi = policy.mlp_extractor.forward_actor(latent_pi)
    distribution = policy._get_action_dist_from_latent(latent_pi)
    log_prob = distribution.log_prob(actions)
    entropy = distribution.entropy()
    mask = episode_starts<1e-8
    return log_prob, entropy, gru_states, mask

    
@dataclasses.dataclass(frozen=True)
class GruBehaviorCloningLossCalculator(BehaviorCloningLossCalculator):
    """Functor to compute the loss used in Behavior Cloning."""

    ent_weight: float
    l2_weight: float

    def __call__(
        self,
        policy: GruActorCriticPolicy,
        obs: Union[
            types.AnyTensor,
            types.DictObs,
            Dict[str, np.ndarray],
            Dict[str, th.Tensor],
        ],
        acts: Union[th.Tensor, np.ndarray],
        gru_states: th.Tensor,
        dones :th.Tensor,
    ) -> GruBCTrainingMetrics:
        tensor_obs = types.map_maybe_dict(
            util.safe_to_tensor,
            types.maybe_unwrap_dictobs(obs),
        )
        acts = util.safe_to_tensor(acts)
        acts = acts.to(policy.device)
        (log_prob, entropy, gru_states, mask) = evaluate_actions(
            policy,
            tensor_obs,  # type: ignore[arg-type]
            acts,
            gru_states,
            dones
        )
        prob_true_act = th.exp(log_prob[mask]).mean()
        log_prob = log_prob[mask].mean()
        entropy = entropy[mask].mean() if entropy is not None else None

        l2_norms = [th.sum(th.square(w)) for w in policy.parameters()]
        l2_norm = sum(l2_norms) / 2  # divide by 2 to cancel with gradient of square
        # sum of list defaults to float(0) if len == 0.
        assert isinstance(l2_norm, th.Tensor)
        ent_loss = -self.ent_weight * (entropy if entropy is not None else th.zeros(1))
        neglogp = -log_prob
        l2_loss = self.l2_weight * l2_norm
        loss = neglogp + ent_loss + l2_loss

        return GruBCTrainingMetrics(
            neglogp=neglogp,
            entropy=entropy,
            ent_loss=ent_loss,
            prob_true_act=prob_true_act,
            l2_norm=l2_norm,
            l2_loss=l2_loss,
            loss=loss,
            gru_states = gru_states
        )

class GRUBC(algo_base.DemonstrationAlgorithm):
    def __init__(
        self,
        *,
        observation_space: gym.Space,
        action_space: gym.Space,
        rng: np.random.Generator,
        policy: Optional[GruActorCriticPolicy] = None,
        demonstrations: Optional[algo_base.AnyTransitions] = None,
        batch_size: int = 32,
        minibatch_size: Optional[int] = None,
        optimizer_cls: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        ent_weight: float = 1e-3,
        l2_weight: float = 0.0,
        device: Union[str, th.device] = "auto",
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    ):
        self._demo_data_loader: Optional[Iterable[types.TransitionMapping]] = None
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size or batch_size
        if self.batch_size % self.minibatch_size != 0:
            raise ValueError("Batch size must be a multiple of minibatch size.")
        super().__init__(
            demonstrations=demonstrations,
            custom_logger=custom_logger,
        )
        self._bc_logger = GruBCLogger(self.logger)

        self.action_space = action_space
        self.observation_space = observation_space

        self.rng = rng
        self._policy = policy.to(utils.get_device(device))
        assert self.policy.observation_space == self.observation_space
        assert self.policy.action_space == self.action_space

        if optimizer_kwargs:
            if "weight_decay" in optimizer_kwargs:
                raise ValueError("Use the parameter l2_weight instead of weight_decay.")
        optimizer_kwargs = optimizer_kwargs or {}
        self.optimizer = optimizer_cls(
            self.policy.parameters(),
            **optimizer_kwargs,
        )

        self.loss_calculator = GruBehaviorCloningLossCalculator(ent_weight, l2_weight)

    @property
    def policy(self) -> GruActorCriticPolicy:
        return self._policy

    def set_demonstrations(self, demonstrations: algo_base.AnyTransitions) -> None:
        self._demo_data_loader = make_data_loader(
            demonstrations,
            self.minibatch_size,
        )

    def train(
        self,
        *,
        n_epochs: Optional[int] = None,
        n_batches: Optional[int] = None,
        on_epoch_end: Optional[Callable[[], None]] = None,
        on_batch_end: Optional[Callable[[], None]] = None,
        log_interval: int = 500,
        log_rollouts_venv: Optional[vec_env.VecEnv] = None,
        log_rollouts_n_episodes: int = 5,
        progress_bar: bool = True,
        reset_tensorboard: bool = False,
    ):
        if reset_tensorboard:
            self._bc_logger.reset_tensorboard_steps()
        self._bc_logger.log_epoch(0)

        compute_rollout_stats = RolloutStatsComputer(
            log_rollouts_venv,
            log_rollouts_n_episodes,
        )

        def _on_epoch_end(epoch_number: int):
            if tqdm_progress_bar is not None:
                total_num_epochs_str = f"of {n_epochs}" if n_epochs is not None else ""
                tqdm_progress_bar.display(
                    f"Epoch {epoch_number} {total_num_epochs_str}",
                    pos=1,
                )
            self._bc_logger.log_epoch(epoch_number + 1)
            if on_epoch_end is not None:
                on_epoch_end()

        mini_per_batch = self.batch_size // self.minibatch_size
        n_minibatches = n_batches * mini_per_batch if n_batches is not None else None

        assert self._demo_data_loader is not None
        demonstration_batches = BatchIteratorWithEpochEndCallback(
            self._demo_data_loader,
            n_epochs,
            n_minibatches,
            _on_epoch_end,
        )
        batches_with_stats = enumerate_batches(demonstration_batches)
        tqdm_progress_bar: Optional[tqdm.tqdm] = None

        if progress_bar:
            batches_with_stats = tqdm.tqdm(
                batches_with_stats,
                unit="batch",
                total=n_minibatches,
            )
            
            tqdm_progress_bar = batches_with_stats

        def process_batch():
            self.optimizer.step()
            self.optimizer.zero_grad()

            if batch_num % log_interval == 0:
                rollout_stats = compute_rollout_stats(self.policy, self.rng)
                self._bc_logger.log_batch(
                    batch_num,
                    minibatch_size,
                    num_samples_so_far,
                    training_metrics,
                    rollout_stats,
                )

            if on_batch_end is not None:
                on_batch_end()
        gru_states = np.concatenate([np.zeros(self.policy.gru_hidden_state_shape) for _ in range(self.batch_size)], axis=1)
        gru_states = util.safe_to_tensor(gru_states, device=self.policy.device).float()
        self.policy.set_training_mode(True)
        self.optimizer.zero_grad()
        for (
            batch_num,
            minibatch_size,
            num_samples_so_far,
        ), batch in batches_with_stats:
            obs_tensor: Union[th.Tensor, Dict[str, th.Tensor]]
            obs_tensor = types.map_maybe_dict(
                lambda x: util.safe_to_tensor(x, device=self.policy.device),
                types.maybe_unwrap_dictobs(batch["obs"]),
            )
            acts = util.safe_to_tensor(batch["acts"], device=self.policy.device)
            dones = util.safe_to_tensor(batch['dones'].float(), device=self.policy.device)
            training_metrics = self.loss_calculator(self.policy, obs_tensor, acts, gru_states, dones)
            loss = training_metrics.loss * minibatch_size / self.batch_size
            loss.backward()

            batch_num = batch_num * self.minibatch_size // self.batch_size
            if num_samples_so_far % self.batch_size == 0:
                process_batch()
        if num_samples_so_far % self.batch_size != 0:
            batch_num += 1
            process_batch()
