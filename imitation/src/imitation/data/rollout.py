import collections
import dataclasses
import logging
from typing import Callable, Dict, Hashable, List, Optional, Sequence, Union
import pdb

import numpy as np
import torch
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import VecEnv

from imitation.data import types
from imitation.util import reward_wrapper, util, video_wrapper
from imitation.util.reward_wrapper import RewardVecEnvWrapper
from imitation.rewards.serialize import load_reward


def unwrap_traj(traj: types.TrajectoryWithRew) -> types.TrajectoryWithRew:
    """Uses `RolloutInfoWrapper`-captured `obs` and `rews` to replace fields.

    This can be useful for bypassing other wrappers to retrieve the original
    `obs` and `rews`.

    Fails if `infos` is None or if the trajectory was generated from an
    environment without imitation.util.rollout.RolloutInfoWrapper

    Args:
      traj: A trajectory generated from `RolloutInfoWrapper`-wrapped Environments.

    Returns:
      A copy of `traj` with replaced `obs` and `rews` fields.
    """
    ep_info = traj.infos[-1]["rollout"]
    res = dataclasses.replace(traj, obs=ep_info["obs"], rews=ep_info["rews"])
    assert len(res.obs) == len(res.acts) + 1
    assert len(res.rews) == len(res.acts)
    return res


class TrajectoryAccumulator:
    """Accumulates trajectories step-by-step.

    Useful for collecting completed trajectories while ignoring partially-completed
    trajectories (e.g. when rolling out a VecEnv to collect a set number of
    transitions). Each in-progress trajectory is identified by a 'key', which enables
    several independent trajectories to be collected at once. They key can also be left
    at its default value of `None` if you only wish to collect one trajectory.
    """

    def __init__(self):
        """Initialise the trajectory accumulator."""
        self.partial_trajectories = collections.defaultdict(list)

    def add_step(self, step_dict: Dict[str, np.ndarray], key: Hashable = None):
        """Add a single step to the partial trajectory identified by `key`.

        Generally a single step could correspond to, e.g., one environment managed
        by a VecEnv.

        Args:
            step_dict: dictionary containing information for the current step. Its
                keys could include any (or all) attributes of a `TrajectoryWithRew`
                (e.g. "obs", "acts", etc.).
            key: key to uniquely identify the trajectory to append to, if working
                with multiple partial trajectories.
        """
        self.partial_trajectories[key].append(step_dict)

    def finish_trajectory(self, key: Hashable = None) -> types.TrajectoryWithRew:
        """Complete the trajectory labelled with `key`.

        Args:
            key: key uniquely identifying which in-progress trajectory to remove.

        Returns:
            traj: list of completed trajectories popped from
                `self.partial_trajectories`.
        """
        part_dicts = self.partial_trajectories[key]
        del self.partial_trajectories[key]
        out_dict_unstacked = collections.defaultdict(list)
        for part_dict in part_dicts:
            for key, array in part_dict.items():
                out_dict_unstacked[key].append(array)
        out_dict_stacked = {
            key: np.stack(arr_list, axis=0)
            for key, arr_list in out_dict_unstacked.items()
        }
        traj = types.TrajectoryWithRew(**out_dict_stacked)
        assert traj.rews.shape[0] == traj.acts.shape[0] == traj.obs.shape[0] - 1
        return traj

    def add_steps_and_auto_finish(
        self,
        acts: np.ndarray,
        obs: np.ndarray,
        rews: np.ndarray,
        dones: np.ndarray,
        infos: List[dict],
    ) -> List[types.TrajectoryWithRew]:
        """Calls `add_step` repeatedly using acts and the returns from `venv.step`.

        Also automatically calls `finish_trajectory()` for each `done == True`.
        Before calling this method, each environment index key needs to be
        initialized with the initial observation (usually from `venv.reset()`).

        See the body of `util.rollout.generate_trajectory` for an example.

        Args:
            acts: Actions passed into `VecEnv.step()`.
            obs: Return value from `VecEnv.step(acts)`.
            rews: Return value from `VecEnv.step(acts)`.
            dones: Return value from `VecEnv.step(acts)`.
            infos: Return value from `VecEnv.step(acts)`.
        Returns:
            A list of completed trajectories. There should be one trajectory for
            each `True` in the `dones` argument.
        """
        trajs = []
        for env_idx in range(len(obs)):
            assert env_idx in self.partial_trajectories
            assert list(self.partial_trajectories[env_idx][0].keys()) == ["obs"], (
                "Need to first initialize partial trajectory using "
                "self._traj_accum.add_step({'obs': ob}, key=env_idx)"
            )

        zip_iter = enumerate(zip(acts, obs, rews, dones, infos))
        # pdb.set_trace()
        for env_idx, (act, ob, rew, done, info) in zip_iter:
            if done and "terminal_observation" in info:
                # actual obs is inaccurate, so we use the one inserted into step info
                # by stable baselines wrapper
                real_ob = info["terminal_observation"]
            else:
                real_ob = ob

            self.add_step(
                dict(
                    acts=act,
                    rews=rew,
                    # this is not the obs corresponding to `act`, but rather the obs
                    # *after* `act` (see above)
                    obs=real_ob,
                    infos=info,
                ),
                env_idx,
            )
            if done:
                # finish env_idx-th trajectory
                new_traj = self.finish_trajectory(env_idx)
                trajs.append(new_traj)
                self.add_step(dict(obs=ob), env_idx)
        return trajs


GenTrajTerminationFn = Callable[[Sequence[types.TrajectoryWithRew]], bool]


def min_episodes(n: int) -> GenTrajTerminationFn:
    """Terminate after collecting n episodes of data.

    Arguments:
      n: Minimum number of episodes of data to collect.
         May overshoot if two episodes complete simultaneously (unlikely).

    Returns:
      A function implementing this termination condition.
    """
    assert n >= 1
    return lambda trajectories: len(trajectories) >= n


def min_timesteps(n: int) -> GenTrajTerminationFn:
    """Terminate at the first episode after collecting n timesteps of data.

    Arguments:
      n: Minimum number of timesteps of data to collect.
        May overshoot to nearest episode boundary.

    Returns:
      A function implementing this termination condition.
    """
    assert n >= 1

    def f(trajectories: Sequence[types.TrajectoryWithRew]):
        timesteps = sum(len(t.obs) - 1 for t in trajectories)
        return timesteps >= n

    return f


def make_sample_until(
    n_timesteps: Optional[int], n_episodes: Optional[int]
) -> GenTrajTerminationFn:
    """Returns a termination condition sampling until n_timesteps or n_episodes.

    Arguments:
      n_timesteps: Minimum number of timesteps to sample.
      n_episodes: Number of episodes to sample.

    Returns:
      A termination condition.

    Raises:
      ValueError if both or neither of n_timesteps and n_episodes are set,
      or if either are non-positive.
    """
    if n_timesteps is not None and n_episodes is not None:
        raise ValueError("n_timesteps and n_episodes were both set")
    elif n_timesteps is not None:
        assert n_timesteps > 0
        return min_timesteps(n_timesteps)
    elif n_episodes is not None:
        assert n_episodes > 0
        return min_episodes(n_episodes)
    else:
        raise ValueError("Set at least one of n_timesteps and n_episodes")


def generate_trajectories(
    policy,
    venv: VecEnv,
    sample_until: GenTrajTerminationFn,
    *,
    deterministic_policy: bool = False,
    rng: np.random.RandomState = np.random,
    get_graph: bool = False,
) -> Sequence[types.TrajectoryWithRew]:
    """Generate trajectory dictionaries from a policy and an environment.

    Args:
      policy (BasePolicy or BaseAlgorithm): A stable_baselines3 policy or algorithm
          trained on the gym environment.
      venv: The vectorized environments to interact with.
      sample_until: A function determining the termination condition.
          It takes a sequence of trajectories, and returns a bool.
          Most users will want to use one of `min_episodes` or `min_timesteps`.
      deterministic_policy: If True, asks policy to deterministically return
          action. Note the trajectories might still be non-deterministic if the
          environment has non-determinism!
      rng: used for shuffling trajectories.

    Returns:
      Sequence of trajectories, satisfying `sample_until`. Additional trajectories
      may be collected to avoid biasing process towards short episodes; the user
      should truncate if required.
    """
    get_action = policy.predict
    if isinstance(policy, BaseAlgorithm):
        policy.set_env(venv)

    # Collect rollout tuples.
    trajectories = []
    # accumulator for incomplete trajectories
    trajectories_accum = TrajectoryAccumulator()
    obs = venv.reset()
    # print('obs reset',obs)
    for env_idx, ob in enumerate(obs):
        # Seed with first obs only. Inside loop, we'll only add second obs from
        # each (s,a,r,s') tuple, under the same "obs" key again. That way we still
        # get all observations, but they're not duplicated into "next obs" and
        # "previous obs" (this matters for, e.g., Atari, where observations are
        # really big).
        trajectories_accum.add_step(dict(obs=ob), env_idx)

    # Now, we sample until `sample_until(trajectories)` is true.
    # If we just stopped then this would introduce a bias towards shorter episodes,
    # since longer episodes are more likely to still be active, i.e. in the process
    # of being sampled from. To avoid this, we continue sampling until all epsiodes
    # are complete.
    #
    # To start with, all environments are active.
    active = np.ones(venv.num_envs, dtype=np.bool)
    while np.any(active):
        acts, _ = get_action(obs, deterministic=deterministic_policy)
        if get_graph:
            obs, rews, graphs, dones, infos = venv.step(acts)
        else:
            obs, rews, dones, infos = venv.step(acts)
        # if 'ob' in infos[0]:
        #     print('infos', infos[0]['ob'], infos[0]['rew'], infos[0]['done'])

        # If an environment is inactive, i.e. the episode completed for that
        # environment after `sample_until(trajectories)` was true, then we do
        # *not* want to add any subsequent trajectories from it. We avoid this
        # by just making it never done.
        dones &= active

        new_trajs = trajectories_accum.add_steps_and_auto_finish(
            acts, obs, rews, dones, infos
        )
        # print('new_trajs', new_trajs)
        trajectories.extend(new_trajs)
        # print('trajectories',trajectories)

        if sample_until(trajectories):
            # Termination condition has been reached. Mark as inactive any environments
            # where a trajectory was completed this timestep.
            active &= ~dones

    # Note that we just drop partial trajectories. This is not ideal for some
    # algos; e.g. BC can probably benefit from partial trajectories, too.

    # Each trajectory is sampled i.i.d.; however, shorter episodes are added to
    # `trajectories` sooner. Shuffle to avoid bias in order. This is important
    # when callees end up truncating the number of trajectories or transitions.
    # It is also cheap, since we're just shuffling pointers.
    rng.shuffle(trajectories)

    # Sanity checks.
    for trajectory in trajectories:
        n_steps = len(trajectory.acts)
        # extra 1 for the end
        exp_obs = (n_steps + 1,) + venv.observation_space.shape
        real_obs = trajectory.obs.shape
        assert real_obs == exp_obs, f"expected shape {exp_obs}, got {real_obs}"
        exp_act = (n_steps,) + venv.action_space.shape
        real_act = trajectory.acts.shape
        assert real_act == exp_act, f"expected shape {exp_act}, got {real_act}"
        exp_rew = (n_steps,)
        real_rew = trajectory.rews.shape
        assert real_rew == exp_rew, f"expected shape {exp_rew}, got {real_rew}"

    if get_graph:
        return trajectories, graphs
    else:
        return trajectories


def generate_init_rollout(
    policy,
    venv,
    num_nodes,
    reward_fn,
    trajs,
    query_episode_steps,
    repeat: bool = False,
    aux_reward_fn=None,
    history=None,
    restrict=None,
    vec_normalize=None,
    G: np.ndarray = None,
    all_demos: np.ndarray = None,
    *,
    deterministic_policy: bool = False,
    rng: np.random.RandomState = np.random,
) -> Sequence[types.TrajectoryWithRew]:
    """Initial trajectory based on the initial reward"""

    # planner
    trajectories = policy.select_subgoal_and_plan(
        reward_fn,
        vec_normalize,
        query_episode_steps,
        G=G,
        aux_value_func=aux_reward_fn,
        history=history,
        is_value=True,
        restrict=restrict,
        stochastic=False,
    )
    trajectories_accum = TrajectoryAccumulator()
    steps = 0
    trajectories_accum.add_step(dict(obs=trajectories[0].obs[0]), 0)
    for i, traj in enumerate(trajectories):
        # new_rews_th = reward_fn(traj.obs[1:])
        # new_rews = new_rews_th.cpu().data.numpy()
        T = len(trajectories[i].acts)
        for t in range(T):
            # trajectories[i].rews[t] = new_rews[t]
            trajectories_accum.add_step(
                dict(
                    obs=trajectories[i].obs[t + 1],
                    acts=trajectories[i].acts[t],
                    rews=trajectories[i].rews[t],
                    # done=False,
                    infos=[],  # trajectories[i].infos[t],
                ),
                0,
            )
            steps += 1
    trajectories = [trajectories_accum.finish_trajectory(0)]

    # Sanity checks.
    for trajectory in trajectories:
        n_steps = len(trajectory.acts)
        # extra 1 for the end
        exp_obs = (n_steps + 1,) + venv.observation_space.shape
        real_obs = trajectory.obs.shape
        assert real_obs == exp_obs, f"expected shape {exp_obs}, got {real_obs}"
        exp_act = (n_steps,) + venv.action_space.shape
        real_act = trajectory.acts.shape
        assert real_act == exp_act, f"expected shape {exp_act}, got {real_act}"
        exp_rew = (n_steps,)
        real_rew = trajectory.rews.shape
        assert real_rew == exp_rew, f"expected shape {exp_rew}, got {real_rew}"

    return trajectories, venv


def generate_queries_set_state(
    policy,
    venv,
    num_nodes,
    reward_fn,
    trajs,
    query_episode_steps,
    repeat: bool = False,
    aux_reward_fn=None,
    history=None,
    restrict=None,
    goal=None,
    all_demos: np.ndarray = None,
    *,
    deterministic_policy: bool = False,
    rng: np.random.RandomState = np.random,
) -> Sequence[types.TrajectoryWithRew]:
    """Generate queries"""

    trajectories = policy.select_subgoal(
        reward_fn,
        None,
        query_episode_steps,
        goal=goal,
        aux_value_func=aux_reward_fn,
        history=history,
        is_value=False,
        restrict=restrict,
        stochastic=False,
    )

    return trajectories, venv


def generate_queries_set_state_eval_cost(
    policy,
    venv,
    num_nodes,
    reward_fn,
    trajs,
    query_episode_steps,
    repeat: bool = False,
    aux_reward_fn=None,
    history=None,
    restrict=None,
    goal=None,
    all_demos: np.ndarray = None,
    *,
    deterministic_policy: bool = False,
    rng: np.random.RandomState = np.random,
) -> Sequence[types.TrajectoryWithRew]:
    """Generate goal states in testing environments based given rewards."""

    # planner
    trajectories, rewards = policy.select_subgoal_eval_cost(
        reward_fn,
        None,
        query_episode_steps,
        goal=goal,
        aux_value_func=aux_reward_fn,
        history=history,
        is_value=False,
        restrict=restrict,
        stochastic=False,
    )

    return trajectories, rewards, venv


def generate_queries_random(
    policy,
    venv,
    num_nodes,
    reward_fn,
    trajs,
    query_episode_steps,
    repeat: bool = False,
    aux_reward_fn=None,
    history=None,
    restrict=None,
    goal=None,
    all_demos: np.ndarray = None,
    *,
    deterministic_policy: bool = False,
    rng: np.random.RandomState = np.random,
) -> Sequence[types.TrajectoryWithRew]:

    trajectories = policy.select_subgoal_random(
        reward_fn,
        None,
        query_episode_steps,
        goal=goal,
        aux_value_func=aux_reward_fn,
        history=history,
        is_value=False,
        restrict=restrict,
        stochastic=False,
    )

    return trajectories, venv


def rollout_stats(trajectories: Sequence[types.TrajectoryWithRew]) -> Dict[str, float]:
    """Calculates various stats for a sequence of trajectories.

    Args:
        trajectories: Sequence of trajectories.

    Returns:
        Dictionary containing `n_traj` collected (int), along with episode return
        statistics (keys: `{monitor_,}return_{min,mean,std,max}`, float values)
        and trajectory length statistics (keys: `len_{min,mean,std,max}`, float
        values).

        `return_*` values are calculated from environment rewards.
        `monitor_*` values are calculated from Monitor-captured rewards, and
        are only included if the `trajectories` contain Monitor infos.
    """
    assert len(trajectories) > 0
    out_stats: Dict[str, float] = {"n_traj": len(trajectories)}
    traj_descriptors = {
        "return": np.asarray([sum(t.rews) for t in trajectories]),
        "len": np.asarray([len(t.rews) for t in trajectories]),
    }

    infos_peek = trajectories[0].infos
    if infos_peek is not None and "episode" in infos_peek[-1]:
        monitor_ep_returns = [t.infos[-1]["episode"]["r"] for t in trajectories]
        traj_descriptors["monitor_return"] = np.asarray(monitor_ep_returns)

    stat_names = ["min", "mean", "std", "max"]
    for desc_name, desc_vals in traj_descriptors.items():
        for stat_name in stat_names:
            stat_value: np.generic = getattr(np, stat_name)(desc_vals)
            # Convert numpy type to float or int. The numpy operators always return
            # a numpy type, but we want to return type float. (int satisfies
            # float type for the purposes of static-typing).
            out_stats[f"{desc_name}_{stat_name}"] = stat_value.item()

    for v in out_stats.values():
        assert isinstance(v, (int, float))
    return out_stats


def mean_return(*args, **kwargs) -> float:
    """Find the mean return of a policy.

    Shortcut to call `generate_trajectories` and fetch the `rollout_stats` value for
    `'return_mean'`; see documentation for `generate_trajectories` and `rollout_stats`.
    """
    trajectories = generate_trajectories(*args, **kwargs)
    return rollout_stats(trajectories)["return_mean"]


def flatten_trajectories(trajectories: Sequence[types.Trajectory]) -> types.Transitions:
    """Flatten a series of trajectory dictionaries into arrays.
    Returns observations, actions, next observations, rewards.
    Args:
        trajectories: list of trajectories.
    Returns:
      The trajectories flattened into a single batch of Transitions.
    """
    keys = ["obs", "next_obs", "acts", "dones", "infos"]
    parts = {key: [] for key in keys}
    for traj in trajectories:
        parts["acts"].append(traj.acts)

        obs = traj.obs
        parts["obs"].append(obs[:-1])
        parts["next_obs"].append(obs[1:])

        dones = np.zeros(len(traj.acts), dtype=np.bool)
        dones[-1] = True
        parts["dones"].append(dones)

        if traj.infos is None:
            infos = np.array([{}] * len(traj))
        else:
            infos = traj.infos
        parts["infos"].append(infos)

    cat_parts = {
        key: np.concatenate(part_list, axis=0) for key, part_list in parts.items()
    }
    lengths = set(map(len, cat_parts.values()))
    assert len(lengths) == 1, f"expected one length, got {lengths}"
    return types.Transitions(**cat_parts)


def flatten_trajectories_with_rew(
    trajectories: Sequence[types.TrajectoryWithRew],
) -> types.TransitionsWithRew:
    transitions = flatten_trajectories(trajectories)
    rews = np.concatenate([traj.rews for traj in trajectories])
    return types.TransitionsWithRew(**dataclasses.asdict(transitions), rews=rews)


def generate_transitions(
    policy, venv, n_timesteps: int, *, truncate: bool = True, **kwargs
) -> types.TransitionsWithRew:
    """Generate obs-action-next_obs-reward tuples.

    Args:
      policy (BasePolicy or BaseAlgorithm): A stable_baselines3 policy or
          algorithm, trained on the gym environment.
      venv: The vectorized environments to interact with.
      n_timesteps: The minimum number of timesteps to sample.
      truncate: If True, then drop any additional samples to ensure that exactly
          `n_timesteps` samples are returned.
      **kwargs: Passed-through to generate_trajectories.

    Returns:
      A batch of Transitions. The length of the constituent arrays is guaranteed
      to be at least `n_timesteps` (if specified), but may be greater unless
      `truncate` is provided as we collect data until the end of each episode.
    """
    traj = generate_trajectories(
        policy, venv, sample_until=min_timesteps(n_timesteps), **kwargs
    )
    transitions = flatten_trajectories_with_rew(traj)
    if truncate and n_timesteps is not None:
        as_dict = dataclasses.asdict(transitions)
        truncated = {k: arr[:n_timesteps] for k, arr in as_dict.items()}
        transitions = types.TransitionsWithRew(**truncated)
    return transitions


def rollout_and_save(
    path: str,
    policy: Union[BaseAlgorithm, BasePolicy],
    venv: VecEnv,
    sample_until: GenTrajTerminationFn,
    *,
    unwrap: bool = True,
    exclude_infos: bool = True,
    verbose: bool = True,
    **kwargs,
) -> None:
    """Generate policy rollouts and save them to a pickled list of trajectories.

    The `.infos` field of each Trajectory is set to `None` to save space.

    Args:
      path: Rollouts are saved to this path.
      venv: The vectorized environments.
      sample_until: End condition for rollout sampling.
      unwrap: If True, then save original observations and rewards (instead of
        potentially wrapped observations and rewards) by calling
        `unwrap_traj()`.
      exclude_infos: If True, then exclude `infos` from pickle by setting
        this field to None. Excluding `infos` can save a lot of space during
        pickles.
      verbose: If True, then print out rollout stats before saving.
      deterministic_policy: Argument from `generate_trajectories`.
    """
    trajs = generate_trajectories(policy, venv, sample_until, **kwargs)
    if unwrap:
        trajs = [unwrap_traj(traj) for traj in trajs]
    if exclude_infos:
        trajs = [dataclasses.replace(traj, infos=None) for traj in trajs]
    if verbose:
        stats = rollout_stats(trajs)
        logging.info(f"Rollout stats: {stats}")

    types.save(path, trajs)
    return trajs


def rollout_and_save_query(
    path: str,
    # policy: Union[BaseAlgorithm, BasePolicy],
    query_policy: Union[BaseAlgorithm, BasePolicy],
    venv: VecEnv,
    num_nodes,
    reward_fn,
    env_name,
    num_vec,
    seed,
    parallel,
    log_dir,
    max_episode_steps,
    post_wrappers,
    # reward_type,
    # reward_path,
    expert_trajs,
    sample_until: GenTrajTerminationFn,
    query_episode_steps,
    *,
    unwrap: bool = True,
    exclude_infos: bool = True,
    verbose: bool = True,
    trajs: np.ndarray = None,
    G: np.ndarray = None,
    all_demos: np.ndarray = None,
    repeat: bool = False,
    aux_reward_fn=None,
    history=None,
    vec_normalize=None,
    **kwargs,
) -> None:
    """Generate policy rollouts and save them to a pickled list of trajectories.

    The `.infos` field of each Trajectory is set to `None` to save space.

    Args:
      path: Rollouts are saved to this path.
      venv: The vectorized environments.
      sample_until: End condition for rollout sampling.
      unwrap: If True, then save original observations and rewards (instead of
        potentially wrapped observations and rewards) by calling
        `unwrap_traj()`.
      exclude_infos: If True, then exclude `infos` from pickle by setting
        this field to None. Excluding `infos` can save a lot of space during
        pickles.
      verbose: If True, then print out rollout stats before saving.
      deterministic_policy: Argument from `generate_trajectories`.
    """

    # if trajs is None:
    #     trajs, graph = generate_trajectories(
    #         policy, venv, sample_until, get_graph=True, **kwargs
    #     )
    #     if unwrap:
    #         trajs = [unwrap_traj(traj) for traj in trajs]
    #     if exclude_infos:
    #         trajs = [dataclasses.replace(traj, infos=None) for traj in trajs]
    #     if verbose:
    #         stats = rollout_stats(trajs)
    #         logging.info(f"Rollout stats: {stats}")
    #     return trajs, graph

    # else:
    #     T = len(trajs[0].acts)

    # reward_fn = load_reward(reward_type, reward_path, venv)
    # norm_rew_fn = common.build_norm_reward_fn(
    #     reward_fn=reward_fn, vec_normalize=vec_normalize
    # )

    if trajs is None or len(trajs) == 0:
        queries, new_venv = generate_init_rollout(
            query_policy,
            venv,
            num_nodes,
            reward_fn,
            trajs,
            query_episode_steps,
            vec_normalize=vec_normalize,
            repeat=repeat,
            G=G,
            # all_demos=all_demos,
            **kwargs,
        )
    else:
        queries, new_venv = generate_queries(
            query_policy,
            venv,
            num_nodes,
            reward_fn,
            trajs,
            query_episode_steps,
            repeat=repeat,
            G=G,
            # all_demos=all_demos,
            aux_reward_fn=aux_reward_fn,
            history=history,
            **kwargs,
        )

    print(queries)

    # if unwrap:
    #     queries = [unwrap_traj(traj) for traj in queries]
    # if exclude_infos:
    #     queries = [dataclasses.replace(traj, infos=None) for traj in queries]
    if verbose and queries is not None:
        stats = rollout_stats(queries)
        logging.info(f"Rollout stats: {stats}")

    # types.save(path, trajs)
    return queries, new_venv


def rollout_and_save_query_set_state(
    path: str,
    # policy: Union[BaseAlgorithm, BasePolicy],
    query_policy: Union[BaseAlgorithm, BasePolicy],
    venv: VecEnv,
    num_nodes,
    reward_fn,
    env_name,
    num_vec,
    seed,
    parallel,
    log_dir,
    max_episode_steps,
    post_wrappers,
    # reward_type,
    # reward_path,
    expert_trajs,
    sample_until: GenTrajTerminationFn,
    query_episode_steps,
    *,
    unwrap: bool = True,
    exclude_infos: bool = True,
    verbose: bool = True,
    trajs: np.ndarray = None,
    goal: None,
    all_demos: np.ndarray = None,
    repeat: bool = False,
    aux_reward_fn=None,
    history=None,
    vec_normalize=None,
    **kwargs,
) -> None:
    """Generate policy rollouts and save them to a pickled list of trajectories.

    The `.infos` field of each Trajectory is set to `None` to save space.

    Args:
      path: Rollouts are saved to this path.
      venv: The vectorized environments.
      sample_until: End condition for rollout sampling.
      unwrap: If True, then save original observations and rewards (instead of
        potentially wrapped observations and rewards) by calling
        `unwrap_traj()`.
      exclude_infos: If True, then exclude `infos` from pickle by setting
        this field to None. Excluding `infos` can save a lot of space during
        pickles.
      verbose: If True, then print out rollout stats before saving.
      deterministic_policy: Argument from `generate_trajectories`.
    """

    queries, new_venv = generate_queries_set_state(
        query_policy,
        venv,
        num_nodes,
        reward_fn,
        trajs,
        query_episode_steps,
        repeat=repeat,
        goal=goal,
        # all_demos=all_demos,
        aux_reward_fn=aux_reward_fn,
        history=history,
        **kwargs,
    )

    print(queries)

    # if unwrap:
    #     queries = [unwrap_traj(traj) for traj in queries]
    # if exclude_infos:
    #     queries = [dataclasses.replace(traj, infos=None) for traj in queries]
    # if verbose and queries is not None:
    #     stats = rollout_stats(queries)
    #     logging.info(f"Rollout stats: {stats}")
    # types.save(path, trajs)
    return queries, new_venv


def rollout_and_save_query_set_state_eval(
    path: str,
    # policy: Union[BaseAlgorithm, BasePolicy],
    query_policy: Union[BaseAlgorithm, BasePolicy],
    venv: VecEnv,
    num_nodes,
    reward_fn,
    env_name,
    num_vec,
    seed,
    parallel,
    log_dir,
    max_episode_steps,
    post_wrappers,
    # reward_type,
    # reward_path,
    expert_trajs,
    sample_until: GenTrajTerminationFn,
    query_episode_steps,
    *,
    unwrap: bool = True,
    exclude_infos: bool = True,
    verbose: bool = True,
    trajs: np.ndarray = None,
    goal: None,
    all_demos: np.ndarray = None,
    repeat: bool = False,
    aux_reward_fn=None,
    history=None,
    vec_normalize=None,
    **kwargs,
) -> None:
    """Generate policy rollouts and save them to a pickled list of trajectories.

    The `.infos` field of each Trajectory is set to `None` to save space.

    Args:
      path: Rollouts are saved to this path.
      venv: The vectorized environments.
      sample_until: End condition for rollout sampling.
      unwrap: If True, then save original observations and rewards (instead of
        potentially wrapped observations and rewards) by calling
        `unwrap_traj()`.
      exclude_infos: If True, then exclude `infos` from pickle by setting
        this field to None. Excluding `infos` can save a lot of space during
        pickles.
      verbose: If True, then print out rollout stats before saving.
      deterministic_policy: Argument from `generate_trajectories`.
    """

    queries, rewards, new_venv = generate_queries_set_state_eval(
        query_policy,
        venv,
        num_nodes,
        reward_fn,
        trajs,
        query_episode_steps,
        repeat=repeat,
        goal=goal,
        # all_demos=all_demos,
        aux_reward_fn=aux_reward_fn,
        history=history,
        **kwargs,
    )

    print(queries)

    # if unwrap:
    #     queries = [unwrap_traj(traj) for traj in queries]
    # if exclude_infos:
    #     queries = [dataclasses.replace(traj, infos=None) for traj in queries]
    # if verbose and queries is not None:
    #     stats = rollout_stats(queries)
    #     logging.info(f"Rollout stats: {stats}")
    # types.save(path, trajs)
    return queries, rewards, new_venv


def rollout_and_save_query_set_state_eval_cost(
    path: str,
    # policy: Union[BaseAlgorithm, BasePolicy],
    query_policy: Union[BaseAlgorithm, BasePolicy],
    venv: VecEnv,
    num_nodes,
    reward_fn,
    env_name,
    num_vec,
    seed,
    parallel,
    log_dir,
    max_episode_steps,
    post_wrappers,
    # reward_type,
    # reward_path,
    expert_trajs,
    sample_until: GenTrajTerminationFn,
    query_episode_steps,
    *,
    unwrap: bool = True,
    exclude_infos: bool = True,
    verbose: bool = True,
    trajs: np.ndarray = None,
    goal: None,
    all_demos: np.ndarray = None,
    repeat: bool = False,
    aux_reward_fn=None,
    history=None,
    vec_normalize=None,
    **kwargs,
) -> None:
    """Generate policy rollouts and save them to a pickled list of trajectories.

    The `.infos` field of each Trajectory is set to `None` to save space.

    Args:
      path: Rollouts are saved to this path.
      venv: The vectorized environments.
      sample_until: End condition for rollout sampling.
      unwrap: If True, then save original observations and rewards (instead of
        potentially wrapped observations and rewards) by calling
        `unwrap_traj()`.
      exclude_infos: If True, then exclude `infos` from pickle by setting
        this field to None. Excluding `infos` can save a lot of space during
        pickles.
      verbose: If True, then print out rollout stats before saving.
      deterministic_policy: Argument from `generate_trajectories`.
    """

    queries, rewards, new_venv = generate_queries_set_state_eval_cost(
        query_policy,
        venv,
        num_nodes,
        reward_fn,
        trajs,
        query_episode_steps,
        repeat=repeat,
        goal=goal,
        # all_demos=all_demos,
        aux_reward_fn=aux_reward_fn,
        history=history,
        **kwargs,
    )

    print(queries)

    # if unwrap:
    #     queries = [unwrap_traj(traj) for traj in queries]
    # if exclude_infos:
    #     queries = [dataclasses.replace(traj, infos=None) for traj in queries]
    # if verbose and queries is not None:
    #     stats = rollout_stats(queries)
    #     logging.info(f"Rollout stats: {stats}")
    # types.save(path, trajs)
    return queries, rewards, new_venv


def rollout_and_save_query_set_state_eval_cost_vh(
    path: str,
    # policy: Union[BaseAlgorithm, BasePolicy],
    query_policy: Union[BaseAlgorithm, BasePolicy],
    venv: VecEnv,
    num_nodes,
    reward_fn,
    env_name,
    num_vec,
    seed,
    parallel,
    log_dir,
    max_episode_steps,
    post_wrappers,
    # reward_type,
    # reward_path,
    expert_trajs,
    sample_until: GenTrajTerminationFn,
    query_episode_steps,
    *,
    unwrap: bool = True,
    exclude_infos: bool = True,
    verbose: bool = True,
    trajs: np.ndarray = None,
    goal: None,
    all_demos: np.ndarray = None,
    repeat: bool = False,
    aux_reward_fn=None,
    history=None,
    vec_normalize=None,
    **kwargs,
) -> None:
    """Generate policy rollouts and save them to a pickled list of trajectories.

    The `.infos` field of each Trajectory is set to `None` to save space.

    Args:
      path: Rollouts are saved to this path.
      venv: The vectorized environments.
      sample_until: End condition for rollout sampling.
      unwrap: If True, then save original observations and rewards (instead of
        potentially wrapped observations and rewards) by calling
        `unwrap_traj()`.
      exclude_infos: If True, then exclude `infos` from pickle by setting
        this field to None. Excluding `infos` can save a lot of space during
        pickles.
      verbose: If True, then print out rollout stats before saving.
      deterministic_policy: Argument from `generate_trajectories`.
    """

    queries, rewards, new_venv = generate_queries_set_state_eval_cost_vh(
        query_policy,
        venv,
        num_nodes,
        reward_fn,
        trajs,
        query_episode_steps,
        repeat=repeat,
        goal=goal,
        # all_demos=all_demos,
        aux_reward_fn=aux_reward_fn,
        history=history,
        **kwargs,
    )

    print(queries)

    # if unwrap:
    #     queries = [unwrap_traj(traj) for traj in queries]
    # if exclude_infos:
    #     queries = [dataclasses.replace(traj, infos=None) for traj in queries]
    # if verbose and queries is not None:
    #     stats = rollout_stats(queries)
    #     logging.info(f"Rollout stats: {stats}")
    # types.save(path, trajs)
    return queries, rewards, new_venv


def rollout_and_save_query_random(
    path: str,
    # policy: Union[BaseAlgorithm, BasePolicy],
    query_policy: Union[BaseAlgorithm, BasePolicy],
    venv: VecEnv,
    num_nodes,
    reward_fn,
    env_name,
    num_vec,
    seed,
    parallel,
    log_dir,
    max_episode_steps,
    post_wrappers,
    # reward_type,
    # reward_path,
    expert_trajs,
    sample_until: GenTrajTerminationFn,
    query_episode_steps,
    *,
    unwrap: bool = True,
    exclude_infos: bool = True,
    verbose: bool = True,
    trajs: np.ndarray = None,
    goal: None,
    all_demos: np.ndarray = None,
    repeat: bool = False,
    aux_reward_fn=None,
    history=None,
    vec_normalize=None,
    **kwargs,
) -> None:
    """Generate policy rollouts and save them to a pickled list of trajectories.

    The `.infos` field of each Trajectory is set to `None` to save space.

    Args:
      path: Rollouts are saved to this path.
      venv: The vectorized environments.
      sample_until: End condition for rollout sampling.
      unwrap: If True, then save original observations and rewards (instead of
        potentially wrapped observations and rewards) by calling
        `unwrap_traj()`.
      exclude_infos: If True, then exclude `infos` from pickle by setting
        this field to None. Excluding `infos` can save a lot of space during
        pickles.
      verbose: If True, then print out rollout stats before saving.
      deterministic_policy: Argument from `generate_trajectories`.
    """

    queries, new_venv = generate_queries_random(
        query_policy,
        venv,
        num_nodes,
        reward_fn,
        trajs,
        query_episode_steps,
        repeat=repeat,
        goal=goal,
        # all_demos=all_demos,
        aux_reward_fn=aux_reward_fn,
        history=history,
        **kwargs,
    )

    print(queries)

    # if unwrap:
    #     queries = [unwrap_traj(traj) for traj in queries]
    # if exclude_infos:
    #     queries = [dataclasses.replace(traj, infos=None) for traj in queries]
    # if verbose and queries is not None:
    #     stats = rollout_stats(queries)
    #     logging.info(f"Rollout stats: {stats}")
    # types.save(path, trajs)
    return queries, new_venv
