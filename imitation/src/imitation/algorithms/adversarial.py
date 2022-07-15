import dataclasses
import logging
import os
from typing import Callable, Dict, Iterable, Mapping, Optional, Type, Union

import gym
import numpy as np
import torch as th
import torch.utils.data as th_data
import torch.utils.tensorboard as thboard
import tqdm
import pdb
from stable_baselines3.common import (
    on_policy_algorithm,
    preprocessing,
    vec_env,
    off_policy_algorithm,
)

from imitation.data import (
    buffer,
    types,
    wrappers,
    rollout,
)  # TODO added rollout - order of importing?
from imitation.rewards import common as rew_common
from imitation.rewards import discrim_nets, reward_nets
from imitation.util import logger, reward_wrapper, util
from imitation.planner.planner_v1.planner import Planner


class AdversarialTrainer:
    """Base class for adversarial imitation learning algorithms like GAIL and AIRL."""

    venv: vec_env.VecEnv
    """The original vectorized environment."""

    venv_norm_obs: vec_env.VecEnv
    """Like `self.venv`, but wrapped with `VecNormalize` normalizing the observations.

    These statistics must be saved along with the model."""

    venv_train: vec_env.VecEnv
    """Like `self.venv`, but wrapped with train reward unless in debug mode.

    If `debug_use_ground_truth=True` was passed into the initializer then
    `self.venv_train` is the same as `self.venv`."""

    def __init__(
        self,
        venv: vec_env.VecEnv,
        gen_algo: on_policy_algorithm.OnPolicyAlgorithm,
        discrim: discrim_nets.DiscrimNet,
        expert_data: Union[Iterable[Mapping], types.Transitions],
        expert_batch_size: int,
        n_disc_updates_per_round: int = 2,
        *,
        log_dir: str = "output/",
        normalize_obs: bool = False,
        normalize_reward: bool = True,
        disc_opt_cls: Type[th.optim.Optimizer] = th.optim.Adam,
        disc_opt_kwargs: Optional[Mapping] = None,
        gen_replay_buffer_capacity: Optional[int] = None,
        init_tensorboard: bool = False,
        init_tensorboard_graph: bool = False,
        debug_use_ground_truth: bool = False,
        G_final_state: bool = False,
        beta: float = None,
        return_value: bool = True,
    ):
        """Builds AdversarialTrainer.

        Args:
            venv: The vectorized environment to train in.
            gen_algo: The generator RL algorithm that is trained to maximize
                discriminator confusion. The generator batch size
                `self.gen_batch_size` is inferred from `gen_algo.n_steps`.
            discrim: The discriminator network. This will be moved to the same
                device as `gen_algo`.
            expert_data: Either a `torch.utils.data.DataLoader`-like object or an
                instance of `Transitions` which is automatically converted into a
                shuffled version of the former type.

                If the argument passed is a `DataLoader`, then it must yield batches of
                expert data via its `__iter__` method. Each batch is a dictionary whose
                keys "obs", "acts", "next_obs", and "dones", correspond to Tensor or
                NumPy array values each with batch dimension equal to
                `expert_batch_size`. If any batch dimension doesn't equal
                `expert_batch_size` then a `ValueError` is raised.

                If the argument is a `Transitions` instance, then `len(expert_data)`
                must be at least `expert_batch_size`.
            expert_batch_size: The number of samples in each batch yielded from
                the expert data loader. The discriminator batch size is twice this
                number because each discriminator batch contains a generator sample for
                every expert sample.
            n_discrim_updates_per_round: The number of discriminator updates after each
                round of generator updates in AdversarialTrainer.learn().
            log_dir: Directory to store TensorBoard logs, plots, etc. in.
            normalize_obs: Whether to normalize observations with `VecNormalize`.
            normalize_reward: Whether to normalize rewards with `VecNormalize`.
            disc_opt_cls: The optimizer for discriminator training.
            disc_opt_kwargs: Parameters for discriminator training.
            gen_replay_buffer_capacity: The capacity of the
                generator replay buffer (the number of obs-action-obs samples from
                the generator that can be stored).

                By default this is equal to `self.gen_batch_size`, meaning that we
                sample only from the most recent batch of generator samples.
            init_tensorboard: If True, makes various discriminator
                TensorBoard summaries.
            init_tensorboard_graph: If both this and `init_tensorboard` are True,
                then write a Tensorboard graph summary to disk.
            debug_use_ground_truth: If True, use the ground truth reward for
                `self.train_env`.
                This disables the reward wrapping that would normally replace
                the environment reward with the learned reward. This is useful for
                sanity checking that the policy training is functional.
        """

        assert (
            logger.is_configured()
        ), "Requires call to imitation.util.logger.configure"
        self._global_step = 0
        self._disc_step = 0
        self.n_disc_updates_per_round = n_disc_updates_per_round

        if expert_batch_size <= 0:
            raise ValueError(f"expert_batch_size={expert_batch_size} must be positive.")

        self.expert_batch_size = expert_batch_size
        if isinstance(expert_data, types.Transitions):
            if len(expert_data) < expert_batch_size:
                raise ValueError(
                    "Provided Transitions instance as `expert_data` argument but "
                    "len(expert_data) < expert_batch_size. "
                    f"({len(expert_data)} < {expert_batch_size})."
                )

            self.expert_data_loader = th_data.DataLoader(
                expert_data,
                batch_size=expert_batch_size,
                collate_fn=types.transitions_collate_fn,
                shuffle=True,
                drop_last=True,
            )
        else:
            self.expert_data_loader = expert_data
        # self.expert_data = expert_data
        self._endless_expert_iterator = util.endless_iter(self.expert_data_loader)

        self.debug_use_ground_truth = debug_use_ground_truth
        self.venv = venv
        self.gen_algo = gen_algo
        self._log_dir = log_dir

        # Create graph for optimising/recording stats on discriminator
        self.discrim = discrim.to(self.gen_algo.device)
        self._disc_opt_cls = disc_opt_cls
        self._disc_opt_kwargs = disc_opt_kwargs or {}
        self._init_tensorboard = init_tensorboard
        self._init_tensorboard_graph = init_tensorboard_graph
        self._disc_opt = self._disc_opt_cls(
            self.discrim.parameters(), **self._disc_opt_kwargs
        )

        if self._init_tensorboard:
            logging.info("building summary directory at " + self._log_dir)
            summary_dir = os.path.join(self._log_dir, "summary")
            os.makedirs(summary_dir, exist_ok=True)
            self._summary_writer = thboard.SummaryWriter(summary_dir)

        self.venv_buffering = wrappers.BufferingWrapper(self.venv)
        self.venv_norm_obs = vec_env.VecNormalize(
            self.venv_buffering, norm_reward=False, norm_obs=normalize_obs
        )

        # Transitions to numpy and tensor
        self.all_demos = expert_data.obs[
            None, :, :
        ]  # TODO why Transitions and not list - flatten_trajectories in train_adversarial? 1=n_demos x T x n_ent*n_feat
        if G_final_state:
            self.all_demos = self.all_demos[:, -1, :][
                :, None, :
            ]  # 1=n_demos x 1 x n_ent*n_feat
            # print('G_final_state dim', self.all_demos.shape)
        self.all_demos = self._torchify_array(self.all_demos).float()
        if debug_use_ground_truth:
            # Would use an identity reward fn here, but RewardFns can't see rewards.
            self.venv_wrapped = self.venv_norm_obs
            self.gen_callback = None
        else:
            self.venv_wrapped = reward_wrapper.RewardVecEnvWrapper(
                self.venv_norm_obs,
                self.discrim.predict_reward_train,
                expert_data=self.all_demos,
            )
            self.gen_callback = self.venv_wrapped.make_log_callback()
        self.venv_train = vec_env.VecNormalize(
            self.venv_wrapped, norm_obs=False, norm_reward=normalize_reward
        )
        # in planner this doesn't do anything. norm updates in train_gen (callback to step_wait in vec_normalize sb3)
        # now have to actively update in train_gen
        self.gen_algo.set_env(self.venv_train)

        if gen_replay_buffer_capacity is None:
            gen_replay_buffer_capacity = self.gen_batch_size
        self._gen_replay_buffer = buffer.ReplayBuffer(
            gen_replay_buffer_capacity, self.venv
        )
        self.beta = beta
        self.return_value = return_value

    def _next_expert_batch(self) -> Mapping:
        return next(self._endless_expert_iterator)

    @property
    def gen_batch_size(self) -> int:
        if isinstance(self.gen_algo, Planner):
            return self.gen_algo.n_steps
        return self.gen_algo.n_steps * self.gen_algo.get_env().num_envs

    def train_disc(
        self,
        *,
        expert_samples: Optional[Mapping] = None,
        gen_samples: Optional[Mapping] = None,
        G: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Perform a single discriminator update, optionally using provided samples.

        Args:
            expert_samples: Transition samples from the expert in dictionary form.
                If provided, must contain keys corresponding to every field of the
                `Transitions` dataclass except "infos". All corresponding values can be
                either NumPy arrays or Tensors. Extra keys are ignored. Must contain
                `self.expert_batch_size` samples.

                If this argument is not provided, then `self.expert_batch_size` expert
                samples from `self.expert_data_loader` are used by default.
            gen_samples: Transition samples from the generator policy in same dictionary
                form as `expert_samples`. If provided, must contain exactly
                `self.expert_batch_size` samples. If not provided, then take
                `len(expert_samples)` samples from the generator replay buffer.

        Returns:
           dict: Statistics for discriminator (e.g. loss, accuracy).
        """
        with logger.accumulate_means("disc"):
            # optionally write TB summaries for collected ops
            write_summaries = self._init_tensorboard and self._global_step % 20 == 0

            # compute loss
            batch = self._make_disc_train_batch(
                gen_samples=gen_samples,
                expert_samples=expert_samples,
                G=G,
            )
            # print('adversarial.py train_disc expert_samples', expert_samples)
            # print('adversarial.py train_disc expert_data', self.expert_data)
            disc_logits = self.discrim.logits_gen_is_high(
                batch["state"],
                batch["action"],
                batch["next_state"],
                batch["done"],
                batch["log_policy_act_prob"],
                batch["graph"],
                self.all_demos,
            )
            loss = self.discrim.disc_loss(disc_logits, batch["labels_gen_is_one"])

            # do gradient step
            self._disc_opt.zero_grad()
            loss.backward()
            self._disc_opt.step()
            self._disc_step += 1

            # compute/write stats and TensorBoard data
            with th.no_grad():
                train_stats = rew_common.compute_train_stats(
                    disc_logits, batch["labels_gen_is_one"], loss
                )
            logger.record("global_step", self._global_step)
            for k, v in train_stats.items():
                logger.record(k, v)
            logger.dump(self._disc_step)
            if write_summaries:
                self._summary_writer.add_histogram("disc_logits", disc_logits.detach())

        return train_stats

    def train_gen(
        self,
        total_timesteps: Optional[int] = None,
        learn_kwargs: Optional[Mapping] = None,
    ):
        """Trains the generator to maximize the discriminator loss.

        After the end of training populates the generator replay buffer (used in
        discriminator training) with `self.disc_batch_size` transitions.

        Args:
          total_timesteps: The number of transitions to sample from
            `self.venv_train` during training. By default,
            `self.gen_batch_size`.
          learn_kwargs: kwargs for the Stable Baselines `RLModel.learn()`
            method.
        """

        if total_timesteps is None:
            total_timesteps = self.gen_batch_size
        if learn_kwargs is None:
            learn_kwargs = {}

        if not isinstance(self.gen_algo, Planner):
            with logger.accumulate_means("gen"):
                self.gen_algo.learn(
                    total_timesteps=total_timesteps,
                    reset_num_timesteps=False,
                    callback=self.gen_callback,
                    **learn_kwargs,
                )
            gen_samples = self.venv_buffering.pop_transitions()
        else:  # Planner
            def value_func(sampled_subgoals_th):
                n_points = len(sampled_subgoals_th)
                return self.discrim.predict_reward_test(sampled_subgoals_th, #st
                                    np.array([8] * n_points), #dummy act 'stop'
                                    np.zeros(tuple(list(sampled_subgoals_th.shape))), #dummy next st
                                    np.array([0] * n_points), #dummy rew
                                    all_demos=self.all_demos,
                                    return_value=self.return_value)
            with logger.accumulate_means("gen"):                 
                active_demo_obj_ids = [exp_goal[1] for exp_goal in self.gen_algo.agent.main_env.expert_subgoals]
                gen_samples = self.gen_algo.select_next_state(value_func, self.venv_norm_obs, 
                                                                    total_timesteps, 
                                                                    gt_reward=self.debug_use_ground_truth,
                                                                    restrict=active_demo_obj_ids)
                gen_samples = rollout.flatten_trajectories(gen_samples) #TrajectoryWithRew to Transitions
                #TODO verify input obs array format
                self.venv_norm_obs.obs_rms.update(gen_samples.obs)
        self._global_step += 1
        self._gen_replay_buffer.store(gen_samples)

    def train(
        self,
        total_timesteps: int,
        callback: Optional[Callable[[int], None]] = None,
        G: Optional[np.ndarray] = None,
    ) -> None:
        """Alternates between training the generator and discriminator.

        Every "round" consists of a call to `train_gen(self.gen_batch_size)`,
        a call to `train_disc`, and finally a call to `callback(round)`.

        Training ends once an additional "round" would cause the number of transitions
        sampled from the environment to exceed `total_timesteps`.

        Args:
          total_timesteps: An upper bound on the number of transitions to sample
              from the environment during training.
          callback: A function called at the end of every round which takes in a
              single argument, the round number. Round numbers are in
              `range(total_timesteps // self.gen_batch_size)`.
        """
        n_rounds = total_timesteps // self.gen_batch_size
        assert n_rounds >= 1, (
            "No updates (need at least "
            f"{self.gen_batch_size} timesteps, have only "
            f"total_timesteps={total_timesteps})!"
        )
        for r in tqdm.tqdm(range(0, n_rounds), desc="round"):
            self.train_gen(self.gen_batch_size)
            for _ in range(self.n_disc_updates_per_round):
                self.train_disc()
            if callback:
                callback(r)
            logger.dump(self._global_step)

        # save replay buffer at end of training
        self._gen_replay_buffer.save_buffer(self._log_dir)

    def _torchify_array(self, ndarray: np.ndarray, **kwargs) -> th.Tensor:
        return th.as_tensor(ndarray, device=self.discrim.device(), **kwargs)

    def _torchify_with_space(
        self, ndarray: np.ndarray, space: gym.Space, **kwargs
    ) -> th.Tensor:
        tensor = th.as_tensor(ndarray, device=self.discrim.device(), **kwargs)
        preprocessed = preprocessing.preprocess_obs(
            tensor,
            space,
            # TODO(sam): can I remove "scale" kwarg in DiscrimNet etc.?
            normalize_images=self.discrim.scale,
        )
        return preprocessed

    def _make_disc_train_batch(
        self,
        *,
        gen_samples: Optional[Mapping] = None,
        expert_samples: Optional[Mapping] = None,
        G: np.ndarray = None,
    ) -> dict:
        """Build and return training batch for the next discriminator update.

        Args:
          gen_samples: Same as in `train_disc_step`.
          expert_samples: Same as in `train_disc_step`.
        """
        if expert_samples is None:
            expert_samples = self._next_expert_batch()

        if gen_samples is None:
            if self._gen_replay_buffer.size() == 0:
                raise RuntimeError(
                    "No generator samples for training. Call `train_gen()` first."
                )
            gen_samples = self._gen_replay_buffer.sample(self.expert_batch_size)
            gen_samples = types.dataclass_quick_asdict(gen_samples)

        n_gen = len(gen_samples["obs"])
        n_expert = len(expert_samples["obs"])
        if not (n_gen == n_expert == self.expert_batch_size):
            raise ValueError(
                "Need to have exactly self.expert_batch_size number of expert and "
                "generator samples, each. "
                f"(n_gen={n_gen} n_expert={n_expert} "
                f"expert_batch_size={self.expert_batch_size})"
            )

        # Guarantee that Mapping arguments are in mutable form.
        expert_samples = dict(expert_samples)
        gen_samples = dict(gen_samples)

        # Convert applicable Tensor values to NumPy.
        for field in dataclasses.fields(types.Transitions):
            k = field.name
            if k == "infos":
                continue
            for d in [gen_samples, expert_samples]:
                if isinstance(d[k], th.Tensor):
                    d[k] = d[k].detach().numpy()
        assert isinstance(gen_samples["obs"], np.ndarray)
        assert isinstance(expert_samples["obs"], np.ndarray)

        # Check dimensions.
        n_samples = n_expert + n_gen
        assert n_expert == len(expert_samples["acts"])
        assert n_expert == len(expert_samples["next_obs"])
        assert n_gen == len(gen_samples["acts"])
        assert n_gen == len(gen_samples["next_obs"])

        # Concatenate rollouts, and label each row as expert or generator.
        obs = np.concatenate([expert_samples["obs"], gen_samples["obs"]])
        non_normalized_obs = obs
        acts = np.concatenate([expert_samples["acts"], gen_samples["acts"]])
        next_obs = np.concatenate([expert_samples["next_obs"], gen_samples["next_obs"]])
        dones = np.concatenate([expert_samples["dones"], gen_samples["dones"]])
        labels_gen_is_one = np.concatenate(
            [np.zeros(n_expert, dtype=int), np.ones(n_gen, dtype=int)]
        )
        # Policy and reward network were trained on normalized observations.
        obs = self.venv_norm_obs.normalize_obs(obs)
        next_obs = self.venv_norm_obs.normalize_obs(next_obs)

        # Calculate generator-policy log probabilities.
        with th.no_grad():
            obs_th = th.as_tensor(obs, device=self.gen_algo.device)
            # print('self.gen_algo type', type(self.gen_algo))
            if isinstance(self.gen_algo, on_policy_algorithm.OnPolicyAlgorithm):
                acts_th = th.as_tensor(acts, device=self.gen_algo.device)
                _, log_act_prob_th, _ = self.gen_algo.policy.evaluate_actions(
                    obs_th, acts_th
                )
                log_act_prob = log_act_prob_th.detach().cpu().numpy()
                del acts_th, log_act_prob_th  # unneeded
            elif isinstance(self.gen_algo, off_policy_algorithm.OffPolicyAlgorithm):
                # off policy, calculate act prob
                actions, log_act_prob_th = self.gen_algo.policy.actor.action_log_prob(
                    obs_th
                )  # TODO need to select logs out of acts. this only returns selected action and log of selected action
                log_act_prob = log_act_prob_th.detach().cpu().numpy()
                del log_act_prob_th  # unneeded
            elif isinstance(self.gen_algo, Planner):
                # Planner
                # print('expert_samples',expert_samples)
                # print('gen_samples',gen_samples)
                expert_infos, gen_infos = expert_samples["infos"], gen_samples["infos"]
                # expert_visitations = np.array([expert_info['visitation'] for expert_info in expert_infos])
                # gen_visitations = np.array([gen_info['visitation'] for gen_info in gen_infos])
                # expert_act_idxs = np.array([expert_info["acts_idx"] for expert_info in expert_infos])
                # gen_act_idxs = np.array([gen_info['acts_idx'] for gen_info in gen_infos])
                expert_subgoals = np.array(
                    [expert_info["sampled_subgoals"] for expert_info in expert_infos]
                )  # B x n_samples x n_entities x state_dim
                expert_selected_subgoals_idx = np.array(
                    [expert_info["goal"] for expert_info in expert_infos]
                )
                gen_subgoals = np.array(
                    [gen_info["sampled_subgoals"] for gen_info in gen_infos]
                )  # B x n_samples x n_entities x state_dim
                gen_selected_subgoals_idx = np.array(
                    [gen_info["goal"] for gen_info in gen_infos]
                )

                # visitations = np.concatenate([expert_visitations, gen_visitations])
                # act_idxs = np.concatenate([expert_act_idxs, gen_act_idxs])
                # visitation_prob = np.array([visit[act]/np.sum(visit) for act,visit in zip(act_idxs,visitations)])
                # visitation_log_prob = np.log(visitation_prob)
                # pdb.set_trace()
                subgoals = np.concatenate(
                    [expert_subgoals, gen_subgoals], axis=0
                )  # B x N_samples x state size
                batch_size, n_subgoal_samples = (
                    subgoals.shape[0],
                    subgoals.shape[1],
                )  # 2B, N_samples
                selected_subgoals_idx = np.concatenate(
                    [expert_selected_subgoals_idx, gen_selected_subgoals_idx]
                )
                norm_subgoals = self.venv_norm_obs.normalize_obs(
                    subgoals
                )  # normalize sampled_subgoals_th
                sampled_subgoals_th = th.as_tensor(
                    norm_subgoals, device=self.gen_algo.device
                )
                sampled_subgoals_th = th.reshape(
                    sampled_subgoals_th, (batch_size * n_subgoal_samples, -1)
                )
                n_points = len(sampled_subgoals_th)
                sampled_subgoal_values, _ = self.discrim.predict_reward_test(
                    sampled_subgoals_th,
                    np.array([8] * n_points),
                    np.zeros(sampled_subgoals_th.shape),
                    np.array([0] * n_points),
                    all_demos=self.all_demos,
                    return_value=self.return_value,
                )
                sampled_subgoal_values = np.reshape(
                    sampled_subgoal_values, (batch_size, n_subgoal_samples)
                )
                selected_subgoal_prob = self.beta * np.array(
                    [
                        sampled_subgoal_values[b_idx][sel_idx]
                        for b_idx, sel_idx in enumerate(selected_subgoals_idx)
                    ]
                )
                sampled_subgoals_prob = np.log(
                    np.sum(np.exp(self.beta * sampled_subgoal_values), axis=1)
                )  # should be n_points x 1
                subgoal_log_prob = (
                    selected_subgoal_prob - sampled_subgoals_prob
                )  # .reshape((n_points,))
                # pdb.set_trace()
                # log(pi(a|g,s)*[pi(g|s)/Z]), pi(g|s)~exp(beta*V(g)), Z: normalization term over all g
                # TODO pi(g|s) also depends on our sampling scheme in planner! (ignore for now)
                # log_act_prob = visitation_log_prob + subgoal_log_prob
                log_act_prob = subgoal_log_prob

            del obs_th  # unneeded
        assert len(log_act_prob) == n_samples
        log_act_prob = log_act_prob.reshape((n_samples,))

        if G is not None:
            G_th = self._torchify_array(np.tile(G, (n_samples, 1, 1, 1)))
        else:
            G_th = None
        # pdb.set_trace()
        batch_dict = {
            "state": self._torchify_with_space(obs, self.discrim.observation_space),
            "action": self._torchify_with_space(acts, self.discrim.action_space),
            "next_state": self._torchify_with_space(
                next_obs, self.discrim.observation_space
            ),
            "done": self._torchify_array(dones),
            "labels_gen_is_one": self._torchify_array(labels_gen_is_one),
            "log_policy_act_prob": self._torchify_array(log_act_prob),
            "graph": G_th,
        }

        return batch_dict


class GAIL(AdversarialTrainer):
    def __init__(
        self,
        venv: vec_env.VecEnv,
        expert_data: Union[Iterable[Mapping], types.Transitions],
        expert_batch_size: int,
        gen_algo: on_policy_algorithm.OnPolicyAlgorithm,
        *,
        # FIXME(sam) pass in discrim net directly; don't ask for kwargs indirectly
        discrim_kwargs: Optional[Mapping] = None,
        **kwargs,
    ):
        """Generative Adversarial Imitation Learning.

        Most parameters are described in and passed to `AdversarialTrainer.__init__`.
        Additional parameters that `GAIL` adds on top of its superclass initializer are
        as follows:

        Args:
            discrim_kwargs: Optional keyword arguments to use while constructing the
                DiscrimNetGAIL.

        """
        discrim_kwargs = discrim_kwargs or {}
        discrim = discrim_nets.DiscrimNetGAIL(
            venv.observation_space, venv.action_space, **discrim_kwargs
        )
        super().__init__(
            venv, gen_algo, discrim, expert_data, expert_batch_size, **kwargs
        )


class AIRL(AdversarialTrainer):
    def __init__(
        self,
        venv: vec_env.VecEnv,
        expert_data: Union[Iterable[Mapping], types.Transitions],
        expert_batch_size: int,
        gen_algo: on_policy_algorithm.OnPolicyAlgorithm,
        *,
        # FIXME(sam): pass in reward net directly, not via _cls and _kwargs
        reward_net_cls: Type[reward_nets.RewardNet] = reward_nets.BasicShapedRewardNet,
        reward_net_kwargs: Optional[Mapping] = None,
        discrim_kwargs: Optional[Mapping] = None,
        **kwargs,
    ):
        """Adversarial Inverse Reinforcement Learning.

        Most parameters are described in and passed to `AdversarialTrainer.__init__`.
        Additional parameters that `AIRL` adds on top of its superclass initializer are
        as follows:

        Args:
            reward_net_cls: Reward network constructor. The reward network is part of
                the AIRL discriminator.
            reward_net_kwargs: Optional keyword arguments to use while constructing
                the reward network.
            discrim_kwargs: Optional keyword arguments to use while constructing the
                DiscrimNetAIRL.
        """
        # TODO(shwang): Maybe offer str=>RewardNet conversion like
        #  stable_baselines3 does with policy classes.
        reward_net_kwargs = reward_net_kwargs or {}
        # pdb.set_trace()
        n_entities = (
            venv.get_attr("num_entities")[0]
            if reward_net_cls is reward_nets.GNNShapedRewardNet
            else 0
        )
        state_dim = (
            venv.get_attr("state_dim")[0]
            if reward_net_cls is reward_nets.GNNShapedRewardNet
            else 0
        )
        reward_network = reward_net_cls(
            action_space=venv.action_space,
            observation_space=venv.observation_space,
            device=gen_algo.device,
            # pytype is afraid that we'll directly call RewardNet() which is an abstract
            # class, hence the disable.
            **reward_net_kwargs,  # pytype: disable=not-instantiable
            n_entities=n_entities,
            state_dim=state_dim,
        )
        # print('adversarial AIRL reward_net_kwargs',reward_net_kwargs)

        discrim_kwargs = discrim_kwargs or {}
        discrim = discrim_nets.DiscrimNetAIRL(reward_network, **discrim_kwargs)
        super().__init__(
            venv, gen_algo, discrim, expert_data, expert_batch_size, **kwargs
        )
