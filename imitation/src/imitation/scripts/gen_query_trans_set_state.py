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
from imitation.scripts.config.gen_query_trans_set_state import (
    gen_query_trans_set_state_ex,
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
    print("prev state:", prev_state)
    print("curr state:", curr_state)
    print("oracle feedback:", curr_reward)
    return 1 if gt_goal.reward_full(nodes_state=curr_state) else -1

    prev_reward = gt_goal.reward_full(nodes_state=prev_state)
    curr_reward = gt_goal.reward_full(nodes_state=curr_state)

    print("prev state:", prev_state)
    print("curr state:", curr_state)
    print("prev r:", prev_reward, "curr r:", curr_reward)
    # return -1 if prev_reward > curr_reward + 0.1 else 1
    return -1 if curr_reward < -3.2 else 1

    alpha = np.exp(prev_reward * beta) / (
        np.exp(curr_reward * beta)  # + np.exp(prev_reward * beta)
    )
    c = np.random.sample()
    print("alpha:", alpha, "c:", c, "prev r:", prev_reward, "curr r:", curr_reward)
    return -1 if c < alpha else 1


SHAPE = ["circle", "trapezoid", "square", "triangle", "rectangular"]
COLOR_NAMES = ["blue", "orange", "green", "red", "purlple"]
R = 0.8


def plot_scene_graph(shapes, colors, nodes_state, edges_class, edges_transform, ax, c):
    # line_styles = ["-", "--", ":", "-."]
    edge_colors = [(0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1)]
    num_entities = edges_class.shape[0]
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


@gen_query_trans_set_state_ex.main
def eval_policy(
    _run,
    env_name: str,
    eval_n_timesteps: Optional[int],
    eval_n_episodes: Optional[int],
    init_episode_steps: Optional[int],
    query_episode_steps: Optional[int],
    num_vec: int,
    nsamples_per_ag: int,
    add_new_state: bool,
    exclude: bool,
    alpha,
    w_prev,
    beta,
    num_transforms: int,
    only_add_transforms: bool,
    prev_reward: bool,
    num_iters: int,
    max_num_attempts: int,
    num_opt_steps: int,
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


    TODO:
        3) recompute rewards based on the previous reward func and graph
    """
    # # paramters TODO: add these to the arguments
    # prop_prob_reduce = 0.8
    # max_num_attempts = 1000
    # # alpha = 0.1
    # num_iters = 20

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
        "trans_norm{}_new{}_e{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.png".format(
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

    history_graphs = []
    history_edges_transform = []
    excluded_graphs = []
    excluded_edges_transform = []

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
            radius=6,
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

        query_policy.agent.main_env.set_state(final_expert_agent_states)

        init_traj = query_policy.agent.main_env.get_full_state()
        init_goal_state = [
            init_traj[ent * state_dim : (ent * state_dim) + 2]
            for ent in range(num_entities)
        ]
        curr_goal_state = copy.deepcopy(init_goal_state)
        print("init_traj")
        print(init_goal_state)
        print("init_graph")
        print(init_graph)

        history = []
        history.append(init_goal_state)

        init_goal = GoalGraph(init_graph, predicates, num_transforms=num_transforms)
        curr_goal = copy.deepcopy(init_goal)

        num_entities = init_graph.shape[0]
        max_num_edges = int((num_entities * num_entities - 1) // 2)

        # plotting goal states, graphs, and acceptance
        ncols = 10
        nrows = min(20, int((num_iters * 2 + ncols - 1) // ncols))
        num_subplots = int(ncols * nrows // 2)
        fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=True)
        fig.set_size_inches(ncols * 1.5, nrows * 1.5)
        axs[0, 0].set_title(
            "Predicates:\n"
            + "".join(
                [
                    "#{}: {}({} {}, {} {}) {}\n".format(
                        predicate_id,
                        predicate[0],
                        COLOR_NAMES[colors[predicate[1]]],
                        SHAPE[shapes[predicate[1]]],
                        COLOR_NAMES[colors[predicate[2]]],
                        SHAPE[shapes[predicate[2]]],
                        predicate[3] if predicate[0] == "dist" else "",
                    )
                    for predicate_id, predicate in enumerate(predicates)
                ]
            ),
        )

        print("init goal:", init_goal.edges_class)

        for iter_id in range(0, num_iters):
            print("iter_id", iter_id)

            if iter_id >= 0:
                plot_scene_graph(
                    shapes,
                    colors,
                    curr_goal_state,
                    curr_goal.edges_class,
                    curr_goal.edges_transform,
                    axs[
                        int((iter_id % num_subplots * 2) // ncols),
                        (iter_id % num_subplots * 2) % ncols,
                    ],
                    0,
                )
                fig.savefig(fig_save_path)

                num_edges = int(np.sum(curr_goal.edges_class > 0) // 2)

                num_attempts = 0
                while num_attempts < max_num_attempts:
                    new_goal = copy.deepcopy(curr_goal)
                    u = np.random.sample()
                    if u < prob_trans or iter_id < min_num_transform_iters:
                        edge_list = []
                        for node_id in range(num_entities - 1):
                            for node_id2 in range(node_id + 1, num_entities):
                                if new_goal.edges_class[node_id, node_id2] > 0:
                                    edge_list.append((node_id, node_id2))
                        assert len(edge_list) > 0
                        sampled_edge_idx = np.random.choice(list(range(len(edge_list))))
                        sampled_edge = edge_list[sampled_edge_idx]
                        node_id, node_id2 = sampled_edge[0], sampled_edge[1]

                        new_transorm = np.random.choice(
                            list(range(new_goal.num_transforms))
                        )
                        new_goal.set_transform(node_id, node_id2, new_transorm)

                    else:
                        # sample a new graph
                        u = np.random.sample()
                        reduce_edge_list = []
                        for node_id in range(num_entities - 1):
                            for node_id2 in range(node_id + 1, num_entities):
                                if new_goal.edges_class[node_id, node_id2] > 0:
                                    reduce_edge_list.append((node_id, node_id2))
                        increase_edge_list = []
                        for node_id in range(num_entities - 1):
                            for node_id2 in range(node_id + 1, num_entities):
                                if new_goal.edges_class[node_id, node_id2] < 1:
                                    increase_edge_list.append((node_id, node_id2))

                        if (u < prop_prob_reduce and len(reduce_edge_list) > 1) or len(
                            increase_edge_list
                        ) == 0:  # reduce edge
                            sampled_edge_idx = np.random.choice(
                                list(range(len(reduce_edge_list)))
                            )
                            sampled_edge = reduce_edge_list[sampled_edge_idx]
                            node_id, node_id2 = sampled_edge[0], sampled_edge[1]
                            prev_edge_class = new_goal.edges_class[node_id, node_id2]
                            assert prev_edge_class > 0
                            new_goal.set_edge(node_id, node_id2, 1 - prev_edge_class)
                        else:  # add edge
                            sampled_edge_idx = np.random.choice(
                                list(range(len(increase_edge_list)))
                            )
                            sampled_edge = increase_edge_list[sampled_edge_idx]
                            node_id, node_id2 = sampled_edge[0], sampled_edge[1]
                            prev_edge_class = new_goal.edges_class[node_id, node_id2]
                            assert prev_edge_class < 1
                            new_goal.set_edge(node_id, node_id2, 1 - prev_edge_class)

                        new_goal.update_graph()
                    if (
                        not exclude
                        or retrieve(new_goal, excluded_graphs, excluded_edges_transform)
                        is None
                    ):
                        break
                    num_attempts += 1
                    if num_attempts == max_num_attempts:
                        break
                if num_attempts == max_num_attempts:
                    print("no new hypotheses!")
                    break

            else:
                new_goal = copy.deepcopy(curr_goal)

            print("new graph:", new_goal.edges_class)
            print("new ransform:", new_goal.edges_transform)

            curr_discrim_net = load_network(reward_path)

            prev_iter_id = None
            last_loss = 0
            if prev_iter_id is not None:
                prev_reward_path = os.path.join(
                    log_dir,
                    "reward_trans_norm{}_new{}_e{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
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
                        str(prev_iter_id),
                    )
                    + ".pt",
                )
                discrim_net = load_network(prev_reward_path)
                print("same as iter", prev_iter_id)

            else:
                # torch.load(discrim_net, reward_path)
                print("reward_path:", reward_path)
                discrim_net = load_network(reward_path)

                # use new_goal.graph to finetune the reward function
                # update R (same architecture + new underlying G + regression from first R on demo states)
                optimizer = torch.optim.Adam(discrim_net.parameters(), lr=0.0003)
                mse_loss = nn.MSELoss()
                ranking_loss = nn.MarginRankingLoss()

                G = torch.from_numpy(new_goal.graph).float().to("cuda")

                for t in range(num_opt_steps):  # TODO #train steps, optimizer lr
                    num_expert_states = expert_states.shape[0]
                    random_indices = np.random.choice(
                        num_expert_states, size=expert_batch_size, replace=True
                    )
                    if normalize_state:
                        batch_expert = copy.deepcopy(norm_states[random_indices, :])
                    else:
                        batch_expert = copy.deepcopy(expert_states[random_indices, :])
                    batch_expert_reward_vals = copy.deepcopy(
                        expert_reward_vals[random_indices]
                    )

                    batch_edge_rep = transform(
                        batch_expert,
                        new_goal.edges_transform,
                        global_trans=global_trans,
                    )
                    batch_rewards = batch_expert_reward_vals

                    batch_edge_rep_th = (
                        torch.from_numpy(copy.deepcopy(batch_edge_rep))
                        .float()
                        .to("cuda")
                    )  # BxNxD

                    batch_rewards_th = torch.from_numpy(
                        copy.deepcopy(batch_rewards)
                    ).to("cuda")

                    output = mse_loss(
                        discrim_net(edge_rep=batch_edge_rep_th, G=G), batch_rewards_th
                    )
                    if (t + 1) % 1000 == 0:
                        print("loss:", t + 1, output.data.cpu().numpy())
                        last_loss = output.data.cpu().numpy()

                    if add_new_state and len(negative_states) > 0:

                        num_expert_states = expert_states.shape[0]
                        random_indices = np.random.choice(
                            list(range(positive_start_id, num_expert_states)),
                            size=expert_batch_size,
                            replace=True,
                        )
                        batch_positive_states = copy.deepcopy(
                            expert_states[random_indices, :]
                        )
                        batch_edge_rep_positive = transform(
                            batch_positive_states,
                            new_goal.edges_transform,
                            global_trans=global_trans,
                        )

                        negative_states_np = np.array(negative_states)
                        # print(negative_states_np)
                        num_negative_states = negative_states_np.shape[0]
                        random_indices = np.random.choice(
                            num_negative_states,
                            size=expert_batch_size,
                            replace=True,
                        )
                        batch_negative_states = copy.deepcopy(
                            negative_states_np[random_indices, :]
                        )
                        batch_edge_rep_negative = transform(
                            batch_negative_states,
                            new_goal.edges_transform,
                            global_trans=global_trans,
                        )

                        batch_edge_rep_positive_th = (
                            torch.from_numpy(batch_edge_rep_positive).float().to("cuda")
                        )  # BxNxD

                        batch_edge_rep_negative_th = (
                            torch.from_numpy(batch_edge_rep_negative).float().to("cuda")
                        )  # BxNxD

                        y = torch.ones(expert_batch_size).to("cuda")

                        output += ranking_loss(
                            discrim_net(edge_rep=batch_edge_rep_positive_th, G=G),
                            discrim_net(edge_rep=batch_edge_rep_negative_th, G=G),
                            y,
                        )

                    optimizer.zero_grad()
                    output.backward()
                    torch.nn.utils.clip_grad_norm_(
                        discrim_net.parameters(), 1
                    )  # exploding gradient
                    optimizer.step()

            history_graphs.append(new_goal.graph)
            history_edges_transform.append(new_goal.edges_transform)

            # save new R
            new_reward_path = os.path.join(
                log_dir,
                "reward_trans_norm{}_new{}_e{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
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
                + ".pt",
            )
            print("saving new reward at ", new_reward_path)
            torch.save(discrim_net, new_reward_path)
            new_reward_type = "DiscrimNetR"

            new_graph_path = os.path.join(
                log_dir,
                "graph_trans_norm{}_new{}_e{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
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
            pickle.dump(
                {"graph": new_goal.graph, "transform": new_goal.edges_transform},
                open(new_graph_path, "wb"),
            )

            # generate query for the new graph

            if normalize_state:
                new_reward_fn0 = load_reward(
                    new_reward_type, new_reward_path, venv_query
                )
                new_reward_fn = common.build_norm_reward_fn_no_demos(
                    reward_fn=new_reward_fn0, vec_normalize=vec_normalize
                )
            else:
                new_reward_fn0 = load_reward(
                    new_reward_type, new_reward_path, venv_query
                )
                new_reward_fn = common.build_reward_fn_no_demos(
                    reward_fn=new_reward_fn0, vec_normalize=vec_normalize
                )

            if normalize_state or iter_id == 0:
                prev_reward_fn0 = load_reward(reward_type, reward_path, venv_query)
                prev_reward_fn = common.build_norm_reward_fn_no_demos(
                    reward_fn=prev_reward_fn0, vec_normalize=vec_normalize
                )
            else:
                prev_reward_fn0 = load_reward(reward_type, reward_path, venv_query)
                prev_reward_fn = common.build_reward_fn_no_demos(
                    reward_fn=prev_reward_fn0, vec_normalize=vec_normalize
                )

            prev_state = (
                query_policy.agent.main_env.agent_states
            )  # use if need to set back
            queries, venv_query = rollout.rollout_and_save_query_set_state(
                rollout_save_path,
                # policy,
                query_policy,
                venv_query,
                num_entities,
                new_reward_fn,
                env_name,
                num_vec,
                seed,
                parallel,
                log_dir,
                max_episode_steps,
                post_wrappers,
                expert_trajs,
                sample_until,
                query_episode_steps,
                trajs=None,
                goal=new_goal,
                all_demos=expert_trajs,
                repeat=False,
                aux_reward_fn=(prev_reward_fn, curr_goal.graph),
                history=history,
                restrict=None,
            )
            print("queries")
            print(queries)

            if queries is None:  # no suitbale subgoals
                oracle_feedback = -1
            else:
                best_query = queries[0]

                best_goal_state = [
                    best_query[-1][ent * state_dim : (ent * state_dim) + 2]
                    for ent in range(num_entities)
                ]

                print("curr graph:", curr_goal.edges_class)
                print("curr transform:", curr_goal.edges_transform)

                print("new graph:", new_goal.edges_class)
                print("new ransform:", new_goal.edges_transform)

                history.append(
                    [
                        best_query[-1][ent * state_dim : (ent * state_dim) + 2]
                        for ent in range(num_entities)
                    ]
                )
                # get oracle feedback
                oracle_feedback = oracle(history[-2], history[-1], gt_goal)

            # plot
            plot_scene_graph(
                shapes,
                colors,
                best_goal_state,
                new_goal.edges_class,
                new_goal.edges_transform,
                axs[
                    int((iter_id % num_subplots * 2 + 1) // ncols),
                    (iter_id % num_subplots * 2 + 1) % ncols,
                ],
                oracle_feedback,
            )
            fig.savefig(fig_save_path)

            if oracle_feedback > 0:
                # accept new goal graph proposal
                curr_goal = copy.deepcopy(new_goal)
                reward_path = new_reward_path
                reward_type = new_reward_type
                curr_goal_state = copy.deepcopy(best_goal_state)
                curr_discrim_net = discrim_net

                if add_new_state:
                    expert_states = np.concatenate(
                        (
                            expert_states,
                            np.array([query_policy.agent.main_env.get_full_state()]),
                        ),
                        axis=0,
                    )
                    expert_reward_vals = np.concatenate(
                        (expert_reward_vals, np.array([final_expert_reward])), axis=0
                    )
                    # pdb.set_trace()

            else:
                # move back to the previous goal hypothesis

                if normalize_state:
                    reward_fn0 = load_reward(reward_type, reward_path, venv_query)
                    reward_fn = common.build_norm_reward_fn_no_demos(
                        reward_fn=reward_fn0, vec_normalize=vec_normalize
                    )
                else:
                    reward_fn0 = load_reward(reward_type, reward_path, venv_query)
                    reward_fn = common.build_reward_fn_no_demos(
                        reward_fn=reward_fn0, vec_normalize=vec_normalize
                    )

                negative_states.append(query_policy.agent.main_env.get_full_state())
                query_policy.agent.main_env.set_state(prev_state)
                history.append(curr_goal_state)
                excluded_graphs.append(new_goal.graph.copy())
                excluded_edges_transform.append(new_goal.edges_transform.copy())

            print("accept graph:")
            print(curr_goal.edges_class)
            print("accept transform:")
            print(curr_goal.edges_transform)

            # save history
            history_path = os.path.join(
                log_dir,
                "history_trans_norm{}_new{}_e{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
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
            pickle.dump(
                history,
                open(history_path, "wb"),
            )

            # save additional training data
            train_data_path = os.path.join(
                log_dir,
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
                    str(iter_id),
                )
                + ".pik",
            )
            pickle.dump(
                {
                    "positive_states": expert_states,
                    "negative_states": np.array(negative_states),
                },
                open(train_data_path, "wb"),
            )

            # save accpeted R
            accepted_reward_path = os.path.join(
                log_dir,
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
                    str(iter_id),
                )
                + ".pt",
            )
            torch.save(curr_discrim_net, accepted_reward_path)

            accepted_graph_path = os.path.join(
                log_dir,
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
            pickle.dump(
                {"graph": curr_goal.graph, "transform": curr_goal.edges_transform},
                open(accepted_graph_path, "wb"),
            )

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
    gen_query_trans_set_state_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main_console()
