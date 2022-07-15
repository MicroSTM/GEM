import logging
import os
import os.path as osp
import time
import copy
from typing import Any, Mapping, Optional
import pdb
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pickle
import math
import random
import json
from datetime import datetime

import gym
from sacred.observers import FileStorageObserver
from stable_baselines3.common.vec_env import VecEnvWrapper

import imitation.util.sacred as sacred_util
from imitation.data import rollout, types
from imitation.policies import serialize, base
from imitation.rewards.serialize import (
    load_reward,
    _load_discrim_net,
    load_discrim_R_as_net,
    load_network,
    _load_reward_net_as_fn,
)
from imitation.scripts.config.eval_set_state import (
    eval_set_state_ex,
)
from imitation.util import reward_wrapper, util, video_wrapper
from imitation.util.reward_wrapper import RewardGraphVecEnvWrapper
from imitation.util.reward_graph import GoalGraph
from imitation.util.coord_transform import *
from imitation.rewards import common
from imitation.planner.planner_v1.planner import Planner


class InteractiveRender(VecEnvWrapper):
    """Render the wrapped environment(s) on screen."""

    def __init__(self, venv, fps):
        super().__init__(venv)
        self.render_fps = fps

    def reset(self):
        ob = self.venv.reset()
        self.venv.render()
        return ob

    def step_wait(self):
        ob = self.venv.step_wait()
        if self.render_fps > 0:
            time.sleep(1 / self.render_fps)
        self.venv.render()
        return ob


def video_wrapper_factory(log_dir: str, **kwargs):
    """Returns a function that wraps the environment in a video recorder."""

    def f(env: gym.Env, i: int) -> gym.Env:
        """Wraps `env` in a recorder saving videos to `{log_dir}/videos/{i}`."""
        directory = os.path.join(log_dir, "videos", str(i))
        return video_wrapper.VideoWrapper(env, directory=directory, **kwargs)

    return f


def belongs_to(goal, goal_set):
    N = goal.shape[0]
    for goal_ in goal_set:
        same = True
        for i in range(N - 1):
            for j in range(i + 1, N):
                if goal_[i, j] != goal[i, j]:
                    same = False
                    break
        if same:
            return True
    return False


def retrieve(goal, history_graphs, history_edges_transform):
    for i, (graph, edges_transform) in enumerate(
        zip(history_graphs, history_edges_transform)
    ):
        if (
            np.abs(goal.graph - graph).sum() < 1e-6
            and np.abs(goal.edges_transform - edges_transform).sum() < 1e-6
        ):
            return i
    return None


def oracle(prev_state, curr_state, gt_goal, beta=1.0):
    """simulated oracle feedback using metropolis acceptance ratio"""
    curr_reward = gt_goal.reward_full(nodes_state=curr_state)
    # print("prev state:", prev_state)
    print("curr state:", curr_state)
    # print("oracle feedback:", curr_reward)
    s, r = gt_goal.reward_full(nodes_state=curr_state, percentage=True)
    print("oracle feedback:", s, r)
    return int(s), r


def dist(state1, state2):
    d = 0
    for pos1, pos2 in zip(state1, state2):
        d += np.linalg.norm(np.array(pos1) - np.array(pos2))
    return d


SHAPE = ["circle", "trapezoid", "square", "triangle", "rectangular"]
COLOR = ["red", "blue", "green", "orange", "purlple"]
R = 0.8
c_weight = 0.02


def plot_scene_graph(shapes, colors, nodes_state, edges_class, edges_transform, ax, c):
    edge_colors = [(0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1)]
    num_entities = len(nodes_state)
    ax.clear()
    ax.set_xlim([6, 26])
    ax.set_ylim([2, 22])
    ax.set_aspect(1)
    for i in range(num_entities):
        ag_shape = SHAPE[shapes[i]]
        color = "C{}".format(colors[i])
        # plot shapes in final state
        x, y = nodes_state[i][0], nodes_state[i][1]
        if ag_shape == "circle":
            ax.add_artist(plt.Circle((x, y), R, color=color))
        elif ag_shape == "trapezoid":
            ax.add_artist(
                plt.Polygon(
                    [x, y]
                    + np.array(
                        [[-1.5 * R, -R], [-0.5 * R, R], [0.5 * R, R], [1.5 * R, -R]]
                    ),
                    color=color,
                )
            )
        elif ag_shape == "square":
            ax.add_artist(
                plt.Polygon(
                    [x, y] + np.array([[R, -R], [-R, -R], [-R, R], [R, R]]),
                    color=color,
                )
            )
        elif ag_shape == "triangle":
            ax.add_artist(
                plt.Polygon(
                    [x, y] + np.array([[-R, -R], [R, -R], [0, 2 * R]]),
                    color=color,
                )
            )
        elif ag_shape == "rectangular":
            ax.add_artist(
                plt.Polygon(
                    [x, y]
                    + np.array(
                        [[-R, -0.5 * R], [-R, 0.5 * R], [R, 0.5 * R], [R, -0.5 * R]]
                    ),
                    color=color,
                )
            )

        if edges_class is not None and edges_transform is not None:
            for j in range(i + 1, num_entities):
                if edges_class[i, j]:
                    num_transforms = edges_transform.shape[2]
                    transform_type = 0
                    for k in range(num_transforms):
                        transform_type = (
                            transform_type * 2
                            + edges_transform[i, j, num_transforms - k - 1]
                        )
                    ax.plot(
                        (nodes_state[i][0], nodes_state[j][0]),
                        (nodes_state[i][1], nodes_state[j][1]),
                        # linestyle=line_styles[transform_type],
                        color=edge_colors[transform_type],
                        linewidth=1.0,
                    )

    for pos in ["top", "bottom", "right", "left"]:
        ax.spines[pos].set_edgecolor("k" if c == 0 else ("g" if c == 1 else "r"))


@eval_set_state_ex.main
def eval_policy(
    _run,
    env_name: str,
    eval_n_timesteps: Optional[int],
    eval_n_episodes: Optional[int],
    init_episode_steps: Optional[int],
    query_episode_steps: Optional[int],
    num_vec: int,
    num_test_episodes: int,
    nsamples_per_ag: int,
    add_new_state: bool,
    exclude: bool,
    alpha,
    w_prev,
    beta,
    num_recent_iters: int,
    num_transforms: int,
    only_add_transforms: bool,
    prev_reward: bool,
    num_iters: int,
    max_num_attempts: int,
    global_trans: bool,
    min_num_transform_iters: int,
    prob_trans: float,
    prop_prob_reduce: float,
    expert_batch_size: int,
    parallel: bool,
    render: bool,
    render_fps: int,
    videos: bool,
    video_kwargs: Mapping[str, Any],
    log_dir: str,
    policy_type: str,
    policy_path: str,
    normalize_state: bool,
    early_stopping_thresh: float,
    reward_type: Optional[str] = None,
    reward_dir: Optional[str] = None,
    model_dir: Optional[str] = None,
    replay_buffer_dir: Optional[str] = None,
    rollout_path: Optional[str] = None,
    max_episode_steps: Optional[int] = None,
    rollout_save_path: Optional[str] = None,
    seed_path: Optional[str] = None,
):
    """Rolls a policy out in an environment, collecting statistics.

    Args:
    #   _seed: generated by Sacred.
      seed: provide same as airl seed
      env_name: Gym environment identifier.
      eval_n_timesteps: Minimum number of timesteps to evaluate for. Set exactly
          one of `eval_n_episodes` and `eval_n_timesteps`.
      eval_n_episodes: Minimum number of episodes to evaluate for. Set exactly
          one of `eval_n_episodes` and `eval_n_timesteps`.
      num_vec: Number of environments to run simultaneously.
      parallel: If True, use `SubprocVecEnv` for true parallelism; otherwise,
          uses `DummyVecEnv`.
      max_episode_steps: If not None, then environments are wrapped by
          TimeLimit so that they have at most `max_episode_steps` steps per
          episode.
      render: If True, renders interactively to the screen.
      render_fps: The target number of frames per second to render on screen.
      videos: If True, saves videos to `log_dir`.
      video_kwargs: Keyword arguments passed through to `video_wrapper.VideoWrapper`.
      log_dir: The directory to log intermediate output to, such as episode reward.
      policy_type: A unique identifier for the saved policy,
          defined in POLICY_CLASSES.
      policy_path: A path to the serialized policy.
      reward_type: If specified, overrides the environment reward with
          a reward of this.
      reward_path: If reward_type is specified, the path to a serialized reward
          of `reward_type` to override the environment reward with.

    Returns:
      Return value of `imitation.util.rollout.rollout_stats()`.
    """

    venv_query = gym.make(env_name)

    num_entities = venv_query.num_entities
    shapes = venv_query.shapes
    colors = venv_query.colors
    reset_obs = venv_query.reset()  # TODO ok to reset?
    state_dim = len(reset_obs) // num_entities

    predicates = venv_query.predicates
    gt_graph = np.zeros((num_entities, num_entities, 2))
    gt_graph[:, :, 0] = 1
    for predicate in predicates:
        id1, id2 = predicate[1], predicate[2]
        gt_graph[id1, id2, 0] = 0
        gt_graph[id1, id2, 1] = 1
        gt_graph[id2, id1, 0] = 0
        gt_graph[id2, id1, 1] = 1

    print("gt predicates:", predicates)
    print("gt_graph:", gt_graph)

    # ground truth goal graph
    gt_goal = GoalGraph(
        gt_graph,
        predicates,
        num_transforms=num_transforms,
    )

    if only_add_transforms:
        assert (
            num_transforms * gt_goal.num_entities * (gt_goal.num_entities - 1) / 2
            >= min_num_transform_iters
        )

    fig_dir = os.path.join(log_dir, "figures")
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    fig_save_path = os.path.join(
        fig_dir,
        "eval_norm{}_new{}_e{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.png".format(
            int(normalize_state),
            int(add_new_state),
            int(exclude),
            global_trans,
            prev_reward,
            alpha,
            query_episode_steps,
            prob_trans,
            prop_prob_reduce,
            num_transforms,
            only_add_transforms,
            min_num_transform_iters,
        ),
    )

    fig = plt.figure()

    result_path = os.path.join(
        fig_dir,
        "eval_norm{}_new{}_e{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pik".format(
            int(normalize_state),
            int(add_new_state),
            int(exclude),
            global_trans,
            prev_reward,
            alpha,
            query_episode_steps,
            prob_trans,
            prop_prob_reduce,
            num_transforms,
            only_add_transforms,
            min_num_transform_iters,
        ),
    )
    if seed_path:
        with open(seed_path, "r") as json_file:
            data = json.load(json_file)
            seed = data["seed"]
    else:
        seed = 1

    print("fig_save_path", fig_save_path)
    os.makedirs(log_dir, exist_ok=True)
    sacred_util.build_sacred_symlink(log_dir, _run)

    logging.basicConfig(level=logging.INFO)
    logging.info("Logging to %s", log_dir)
    sample_until = rollout.make_sample_until(eval_n_timesteps, eval_n_episodes)
    post_wrappers = [video_wrapper_factory(log_dir, **video_kwargs)] if videos else None
    print("env_name", env_name)
    venv = util.make_vec_env(
        env_name,
        1,
        seed=seed,
        parallel=parallel,
        log_dir=log_dir,
        max_episode_steps=max_episode_steps,
        post_wrappers=post_wrappers,
    )

    try:

        callback_objs = []
        vec_normalize_path = os.path.join(
            os.path.join(reward_dir, "gen_policy"), "vec_normalize.pkl"
        )
        reward_path = os.path.join(reward_dir, "discrim.pt")
        reward_fn = load_reward(reward_type, reward_path, venv_query)  # reward function
        discrim_net = load_discrim_R_as_net(reward_path)
        with open(vec_normalize_path, "rb") as f:
            vec_normalize = pickle.load(f)
        vec_normalize.training = False

        # TODO load vs accessing base or poten?
        rew_fn = common.build_reward_fn(reward_fn=reward_fn)

        expert_trajs = types.load(rollout_path)
        expert_trajs = expert_trajs[0].obs[None, :, :]

        expert_states = copy.deepcopy(expert_trajs[0])
        n_points = len(expert_states)
        positive_start_id = n_points - 2
        expert_reward_vals, _ = rew_fn(
            expert_states,
            np.array([3] * n_points),
            np.zeros_like(expert_states),
            np.array([0] * n_points),
            all_demos=expert_trajs,
            norm_reward=False,
        )

        print("expert_reward_vals:", expert_reward_vals)
        final_expert_reward = expert_reward_vals[-1]
        final_expert_state = expert_states[-1]
        assert oracle(
            [
                final_expert_state[i * state_dim : i * state_dim + 2]
                for i in range(num_entities)
            ],
            [
                final_expert_state[i * state_dim : i * state_dim + 2]
                for i in range(num_entities)
            ],
            gt_goal,
        )
        final_expert_agent_states = venv_query.convert_fullobs2state(final_expert_state)
        print(final_expert_agent_states, final_expert_reward)

        negative_states = []

        norm_states = vec_normalize.normalize_obs(expert_states)

        random.seed(seed)
        np.random.seed(seed)
        query_policy = Planner(
            env_name,
            log_dir,
            radius=10,
            alpha=alpha,
            beta=beta,
            w_prev=w_prev if prev_reward else 0.0,
            nsamples_per_ag=nsamples_per_ag,
            max_episode_steps=100000,
            main_env=venv_query,
            nb_simulations=500,
        )

        init_graph = np.zeros((num_entities, num_entities, 2))
        for id1 in range(num_entities):
            for id2 in range(num_entities):
                if id1 == id2:
                    init_graph[id1, id2, 0] = 1
                else:
                    init_graph[id1, id2, 1] = 1

        reward_path = os.path.join(
            log_dir,
            "reward_trans_init_norm{}_new{}_e{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pt".format(
                int(normalize_state),
                int(add_new_state),
                int(exclude),
                global_trans,
                prev_reward,
                alpha,
                query_episode_steps,
                prob_trans,
                prop_prob_reduce,
                num_transforms,
                only_add_transforms,
                min_num_transform_iters,
            ),
        )
        print("saving first reward at ", reward_path)
        torch.save(discrim_net, reward_path)  # takes in G and R
        reward_type = "DiscrimNetR"

        init_reward_fn0 = load_reward(reward_type, reward_path, venv_query)
        init_reward_fn = common.build_norm_reward_fn_no_demos(
            reward_fn=init_reward_fn0, vec_normalize=vec_normalize
        )

        def value_func(sampled_subgoals_th):
            n_points = len(sampled_subgoals_th)
            return reward_fn(
                sampled_subgoals_th,  # st
                np.array([8] * n_points),  # dummy act 'stop'
                np.zeros(tuple(list(sampled_subgoals_th.shape))),  # dummy next st
                np.array([0] * n_points),  # dummy rew
                all_demos=expert_trajs,
                return_value=True,
            )

        query_policy.agent.main_env.reset()

        init_traj = query_policy.agent.main_env.get_full_state()
        init_state = [
            init_traj[ent * state_dim : (ent * state_dim) + 2]
            for ent in range(num_entities)
        ]
        curr_state = copy.deepcopy(init_state)

        test_history = []
        test_history.append(init_state)

        init_goal = GoalGraph(init_graph, predicates, num_transforms=num_transforms)
        curr_goal = copy.deepcopy(init_goal)

        max_num_edges = int((num_entities * num_entities - 1) // 2)

        # plotting goal states, graphs, and acceptance

        print("init goal:", init_goal.edges_class)
        prev_goal = copy.deepcopy(init_goal)
        prev_num_edges = int((1 - prev_goal.graph[:, :, 0]).sum() // 2)
        prev_iter = None

        # pdb.set_trace()

        S = []
        R = []
        G = []

        cnt = 0

        for iter_id in range(num_iters):
            print("iter_id", iter_id)

            accepted_graph_path = os.path.join(
                model_dir,
                "accepted_graph_trans_norm{}_new{}_e{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
                    int(normalize_state),
                    int(add_new_state),
                    int(exclude),
                    global_trans,
                    prev_reward,
                    alpha,
                    query_episode_steps,
                    prob_trans,
                    prop_prob_reduce,
                    num_transforms,
                    only_add_transforms,
                    min_num_transform_iters,
                    str(iter_id),
                )
                + ".pik",
            )
            curr_graph = pickle.load(
                open(accepted_graph_path, "rb"),
            )

            curr_goal = GoalGraph(
                curr_graph["graph"], predicates, num_transforms=num_transforms
            )
            curr_goal.edges_transform = curr_graph["transform"].copy()

            curr_num_edges = int((1 - curr_goal.graph[:, :, 0]).sum() // 2)
            changed = False
            if (
                iter_id == 0
                or num_recent_iters == 0
                or (
                    curr_num_edges < prev_num_edges
                    or curr_num_edges == prev_num_edges
                    and cnt > num_recent_iters
                )
            ):
                cnt = 0
                prev_goal = copy.deepcopy(curr_goal)
                prev_iter = iter_id
                curr_iter = iter_id
                prev_num_edges = curr_num_edges
                changed = True
            else:
                curr_goal = copy.deepcopy(prev_goal)
                curr_iter = prev_iter
                curr_num_edges = prev_num_edges
                cnt += 1

            print("curr_iter:", curr_iter)

            restrict = []
            for entity_id in range(num_entities):
                if curr_goal.graph[entity_id, :, 0].sum() < num_entities:
                    restrict.append(entity_id)

            # save additional training data
            train_data_path = os.path.join(
                model_dir,
                "train_data_trans_norm{}_new{}_e{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
                    int(normalize_state),
                    int(add_new_state),
                    int(exclude),
                    global_trans,
                    prev_reward,
                    alpha,
                    query_episode_steps,
                    prob_trans,
                    prop_prob_reduce,
                    num_transforms,
                    only_add_transforms,
                    min_num_transform_iters,
                    str(curr_iter),
                )
                + ".pik",
            )
            curr_train_data = pickle.load(
                open(train_data_path, "rb"),
            )
            positive_states = curr_train_data["positive_states"]
            negative_states = curr_train_data["negative_states"]

            # save accpeted R
            accepted_reward_type = "DiscrimNetR"
            accepted_reward_path = os.path.join(
                model_dir,
                "accepted_reward_trans_norm{}_new{}_e{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
                    int(normalize_state),
                    int(add_new_state),
                    int(exclude),
                    global_trans,
                    prev_reward,
                    alpha,
                    query_episode_steps,
                    prob_trans,
                    prop_prob_reduce,
                    num_transforms,
                    only_add_transforms,
                    min_num_transform_iters,
                    str(curr_iter),
                )
                + ".pt",
            )
            curr_discrim_net = torch.load(accepted_reward_path)

            if normalize_state:
                curr_reward_fn0 = load_reward(
                    accepted_reward_type, accepted_reward_path, venv_query
                )
                curr_reward_fn = common.build_norm_reward_fn_no_demos(
                    reward_fn=curr_reward_fn0, vec_normalize=vec_normalize
                )
            else:
                curr_reward_fn0 = load_reward(
                    accepted_reward_type, accepted_reward_path, venv_query
                )
                curr_reward_fn = common.build_reward_fn_no_demos(
                    reward_fn=curr_reward_fn0, vec_normalize=vec_normalize
                )

            max_reward = -1e6
            added_state = None
            for episode_id in range(num_test_episodes):
                best_query, best_goal_state = None, None
                best_reward = -1e6
                # torch.load(new_reward_net, new_reward_path)
                query_policy.agent.main_env.reset()
                (
                    queries,
                    rewards,
                    venv_query,
                ) = rollout.rollout_and_save_query_set_state_eval_cost(
                    rollout_save_path,
                    # policy,
                    query_policy,
                    venv_query,
                    num_entities,
                    curr_reward_fn,
                    env_name,
                    num_vec,
                    seed,
                    parallel,
                    log_dir,
                    max_episode_steps,
                    post_wrappers,
                    expert_trajs,
                    sample_until,
                    1,
                    trajs=None,
                    goal=curr_goal,
                    all_demos=expert_trajs,
                    repeat=False,
                    aux_reward_fn=None,
                    history=None,
                    restrict=restrict,
                )
                print("queries")
                print(queries)
                print("rewards")
                print(rewards)

                if queries is None:  # no suitbale subgoals
                    oracle_feedback = -1
                else:
                    curr_query = queries[0]
                    curr_reward = rewards[0][-1]

                    curr_goal_state = [
                        curr_query[-1][ent * state_dim : (ent * state_dim) + 2]
                        for ent in range(num_entities)
                    ]

                    if curr_reward > best_reward:
                        best_reward = curr_reward
                        best_goal_state = curr_goal_state
                        best_query = curr_query

                print("curr graph:", curr_goal.edges_class)
                print("curr transform:", curr_goal.edges_transform)
                print("best reward:", best_reward)
                print("best_goal_state:", best_goal_state)

                tmp_state = [
                    best_query[-1][ent * state_dim : (ent * state_dim) + 2]
                    for ent in range(num_entities)
                ]
                # get oracle feedback
                s, _ = oracle(test_history[-1], tmp_state, gt_goal)
                r = s - c_weight * dist(test_history[0], test_history[-1])

                if best_reward > max_reward:
                    max_reward = best_reward
                    success = s
                    reward = r
                    added_state = tmp_state

            test_history.append(added_state)
            S.append((success))
            R.append((reward))
            G.append(copy.deepcopy(curr_goal))

            print("==============")
            print("iter {}:".format(iter_id), S[-1], R[-1])
            print("==============")
            pickle.dump(
                {"S": S, "R": R, "G": G, "test_history": test_history},
                open(result_path, "wb"),
            )

            plt.clf()
            plt.plot(R)
            fig.savefig(fig_save_path)

        fig.savefig(fig_save_path)
    finally:
        # venv.close()
        pass


def main_console():
    # now = datetime.now()
    # dt_string = now.strftime("%d%m%Y%H%M%S")
    # observer = FileStorageObserver(
    #     osp.join("../output", "sacred", "train")
    # )
    # gen_query_trans_set_state_ex.observers.append(observer)
    eval_set_state_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main_console()
