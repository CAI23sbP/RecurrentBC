
import dataclasses
import itertools
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
)

from stable_baselines3.common.vec_env import VecNormalize
from typing import NamedTuple, Generator
from sb3_contrib.common.recurrent.buffers import create_sequencers
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
import gymnasium as gym
import numpy as np
import torch as th
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common import utils
from copy import deepcopy
from imitation.algorithms import base as algo_base
from imitation.data import types
from imitation.util import logger as imit_logger
from imitation.util import util
from .gru_actor_critic import GruActorCriticPolicy

def evaluate_actions(policy: GruActorCriticPolicy, obs: th.Tensor, actions: th.Tensor, gru_states: th.Tensor, episode_starts: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    features = policy.extract_features(obs)
    latent_pi, gru_states = policy._process_sequence(features, gru_states, episode_starts, policy.gru)
    latent_vf = latent_pi.detach()
    
    latent_pi = policy.mlp_extractor.forward_actor(latent_pi)
    latent_vf = policy.mlp_extractor.forward_critic(latent_vf)

    distribution = policy._get_action_dist_from_latent(latent_pi)
    log_prob = distribution.log_prob(actions)
    values = policy.value_net(latent_vf)
    return values, log_prob, distribution.entropy(), gru_states
import logging
from typing import Any

# Configure the Python logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclasses.dataclass(frozen=True)
class BCTrainingMetrics:
    """Container for the different components of behavior cloning loss."""

    neglogp: th.Tensor
    entropy: Optional[th.Tensor]
    ent_loss: th.Tensor  # set to 0 if entropy is None
    prob_true_act: th.Tensor
    l2_norm: th.Tensor
    l2_loss: th.Tensor
    loss: th.Tensor

class BCLogger:
    """Utility class to help logging information relevant to Behavior Cloning."""

    def __init__(self, logger: imit_logger.HierarchicalLogger):
        """Create new BC logger.

        Args:
            logger: The logger to feed all the information to.
        """
        self._logger = logger
        self._tensorboard_step = 0
        self._current_epoch = 0

    def reset_tensorboard_steps(self):
        self._tensorboard_step = 0

    def log_epoch(self, epoch_number):
        self._current_epoch = epoch_number

    def log_batch(
        self,
        batch_num: int,
        batch_size: int,
        training_metrics: BCTrainingMetrics,
    ):
        self._logger.record("batch_size", batch_size)
        self._logger.record("bc/epoch", self._current_epoch)
        self._logger.record("bc/batch", batch_num)
        for k, v in training_metrics.__dict__.items():
            self._logger.record(f"bc/{k}", float(v) if v is not None else None)
        logging.info('-'*10)
        logging.info(f"Epoch: {self._current_epoch}, Batch: {batch_num}, Batch Size: {batch_size}")
        for key, value in training_metrics.__dict__.items():
            logging.info(f"{key}: {value}")
        logging.info('-'*10)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_logger"]
        return state
    
class GruBC(algo_base.DemonstrationAlgorithm):
    """
    rollout_size_init
    hidden = zeors 
    for data in demon:
        hidden = policy.forward(obs)
        rollout.add(*data + hidden)
    """

    """
    You must know that there didn't use random mechanism to make batch, it means we can use it right now for recurrent!.
        i.e. We don't need to make batch for recurrent
    But how to recurrent a hidden state?

    Facebook, they used batch for make hidden state, here is a pseudocode
        init empty list for stack hidden_state as A
        for batch in data_generator:
            _, log_pros, hidden_state, dist_entropy = self.policy.evaluated_actions(batch[obs],
                                                                                    batch[acts],
                                                                                    batch['hidden_state'])
            // skipping change above outputs' shape 
            input hidden_state into A
        concat A and transform numpy as torch
        return A
    
    Here is one question
        1. Here is first hidden state? -> In my opinion, we should add this in BehaviorCloningLossCalculator at __init__ part
    Note
        1. Don't think of episode start tensor, i will add this part. 
        2. Don't consider about reset hidden state, In _process_sequence from Actor-Critic will be reset hidden state by using episode_starts
    """
        
    def __init__(
        self,
        *,
        venv: VecNormalize,
        rng: np.random.Generator,
        policy: GruActorCriticPolicy ,
        demonstrations: Optional[algo_base.AnyTransitions] = None,
        batch_size: int = 32,
        minibatch_size: Optional[int] = None,
        optimizer_cls: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        ent_weight: float = 1e-3,
        l2_weight: float = 0.0,
        device: Union[str, th.device] = "auto",
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        tensorboard_log: str,
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
        self._bc_logger = BCLogger(self.logger)
        self.buffer_size = batch_size
        self.ent_weight = ent_weight
        self.l2_weight = l2_weight
        self._policy = policy.to(utils.get_device(device))
        self.writer = SummaryWriter(tensorboard_log)
        self.buffer = BCBuffer(observation_space= venv.observation_space,
                              action_space=venv.action_space,
                              policy= self.policy,
                              buffer_size=self.buffer_size, 
                              n_envs = 1)

        self.single_hidden_state_shape = (policy.gru.num_layers, 1, policy.gru.hidden_size)
        optimizer_kwargs = optimizer_kwargs or {}
        self.optimizer = optimizer_cls(
            self.policy.parameters(),
            **optimizer_kwargs,
        )
        self.current_epoch = 0
    @property
    def policy(self) -> GruActorCriticPolicy:
        return self._policy

    def set_demonstrations(self, demonstrations: algo_base.AnyTransitions) -> None:
        self.demonstrations = demonstrations
    
    def train(
        self,
        *,
        n_epochs: Optional[int] = None,
        n_batches: Optional[int] = None,
        log_interval: int = 500,

    ):
        self.interation = 0
        self.buffer.reset()
        self.policy.train()
        self._bc_logger.reset_tensorboard_steps()
        self._bc_logger.log_epoch(0)

        self._last_gru_states = th.zeros(self.single_hidden_state_shape, device=self.policy.device)
        gru_states = deepcopy(self._last_gru_states)
        for index ,demon_info in enumerate(self.demonstrations):
            obs = util.safe_to_tensor(demon_info['obs'], device=self.policy.device)[None]
            acts = util.safe_to_tensor(demon_info['acts'], device=self.policy.device)[None]
            dones = util.safe_to_tensor(demon_info['dones'], device=self.policy.device).to(dtype=th.float)[None]
            self.buffer.insert(demon_info['obs'][None], demon_info['acts'][None], gru_states, demon_info['dones'][None])
            _, _, _, gru_states = self.policy.forward(obs, gru_states, dones)
            self._last_gru_states = gru_states
            if (index +1) % self.buffer_size == 0 :
                self.learn(n_epochs, n_batches, log_interval)
                
        self.writer.close()

    def learn(self, n_epochs, n_batches, log_interval):
        for epoch in range(n_epochs):
            self.current_epoch += epoch
            self._bc_logger.log_epoch(epoch + 1)
            for batch_num ,rollout_data in enumerate(self.buffer.get(n_batches)):
                _ ,log_prob, entropy, _ = evaluate_actions(self.policy,
                                                            rollout_data.observations,
                                                            rollout_data.actions,
                                                            rollout_data.recurrent_hidden_states,
                                                            rollout_data.episode_starts)
                prob_true_act = th.exp(log_prob).mean()
                log_prob = log_prob.mean()
                
                entropy = entropy.mean() if entropy is not None else None
                l2_norms = [th.sum(th.square(w)) for w in self.policy.parameters()]
                l2_norm = sum(l2_norms) / 2  # divide by 2 to cancel with gradient of square
                ent_loss = -self.ent_weight * (entropy if entropy is not None else th.zeros(1))
                neglogp = -log_prob
                l2_loss = self.l2_weight * l2_norm
                loss = neglogp + ent_loss + l2_loss
                training_metrics = BCTrainingMetrics(neglogp=neglogp, 
                                  entropy=entropy,
                                  prob_true_act=prob_true_act,
                                  ent_loss=ent_loss,
                                  l2_norm=l2_norm,
                                  l2_loss=l2_loss,
                                  loss=loss
                                  )
                if self.current_epoch % log_interval == 0:
                    self._bc_logger.log_batch(
                        batch_num = batch_num,
                        batch_size = self.minibatch_size,
                        training_metrics= training_metrics,
                    )
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.writer.add_scalar('GRU_BC/Loss',loss.item(), self.interation)
                self.interation += 1

class GruRolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    recurrent_hidden_states: th.Tensor
    episode_starts: th.Tensor

class BCBuffer():
    def __init__(self, observation_space:gym.spaces, action_space:gym.spaces, policy, buffer_size: int, n_envs: int):
        self.observation_space = observation_space
        self.obs_shape = get_obs_shape(observation_space)
        self.action_dim = get_action_dim(action_space)
        gru = policy.gru
        self.device  = policy.device
        self.n_envs = n_envs
        self.hidden_state_shape = (buffer_size, gru.num_layers, self.n_envs, gru.hidden_size)
        self.buffer_size = buffer_size
        self.pos = 0
        self.recu_pos = 0
        self.full = False
        self.recu_full = False
        self.generator_ready = False

    def reset(self):
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.recurrent_hidden_states = np.zeros(self.hidden_state_shape, dtype=np.float32)
        self.pos = 0
        self.full = False
        self.generator_ready = False

    def insert(self, obs, action, hidden_state, episode_starts):
        if isinstance(self.observation_space, gym.spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        action = action.reshape((self.n_envs, self.action_dim))
        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(action)
        self.episode_starts[self.pos] = np.array(episode_starts)
        self.recurrent_hidden_states[self.pos] = np.array(hidden_state.detach().cpu().numpy())
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def get(self, batch_size: Optional[int] = None) -> Generator[GruRolloutBufferSamples, None, None]:
        assert self.full, "Rollout buffer must be full before sampling from it"
        # Prepare the data
        if not self.generator_ready:
            self.__dict__['recurrent_hidden_states'] = self.__dict__['recurrent_hidden_states'].swapaxes(1, 2)

            for tensor in [
                "observations",
                "actions",
                "recurrent_hidden_states",
                "episode_starts",
            ]:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor]) #
            self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        split_index = np.random.randint(self.buffer_size * self.n_envs)
        indices = np.arange(self.buffer_size * self.n_envs)
        indices = np.concatenate((indices[split_index:], indices[:split_index]))
        env_change = np.zeros(self.buffer_size * self.n_envs).reshape(self.buffer_size, self.n_envs)
        # Flag first timestep as change of environment
        env_change[0, :] = 1.0
        env_change = self.swap_and_flatten(env_change)
        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            batch_inds = indices[start_idx : start_idx + batch_size]
            yield self._get_samples(batch_inds, env_change)
            start_idx += batch_size

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return th.tensor(array, device=self.device)
        return th.as_tensor(array, device=self.device)


    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env_change: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> GruRolloutBufferSamples:
        # Retrieve sequence starts and utility function
        self.seq_start_indices, self.pad, self.pad_and_flatten = create_sequencers(
            self.episode_starts[batch_inds], env_change[batch_inds], self.device
        )

        # Number of sequences
        n_seq = len(self.seq_start_indices)
        max_length = self.pad(self.actions[batch_inds]).shape[1]
        padded_batch_size = n_seq * max_length
        # We retrieve the gru hidden states that will allow
        # to properly initialize the GRU at the beginning of each sequence
        recurrent_hidden_states = self.recurrent_hidden_states[batch_inds][self.seq_start_indices].swapaxes(0, 1)
        recurrent_hidden_states = self.to_torch(recurrent_hidden_states).contiguous()

        return GruRolloutBufferSamples(
            # (batch_size, obs_dim) -> (n_seq, max_length, obs_dim) -> (n_seq * max_length, obs_dim)
            observations=self.pad(self.observations[batch_inds]).reshape((padded_batch_size, *self.obs_shape)),
            actions=self.pad(self.actions[batch_inds]).reshape((padded_batch_size,) + self.actions.shape[1:]),
            recurrent_hidden_states= recurrent_hidden_states,
            episode_starts=self.pad_and_flatten(self.episode_starts[batch_inds]),
        )
