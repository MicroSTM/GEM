from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import gym
from types import SimpleNamespace
import torch
import argparse
import pdb
import copy
from datetime import datetime

from imitation.planner.planner_v1.MCTS_agent import *

import imitation.util.sacred as sacred_util
from imitation.data import rollout, types
from imitation.policies import serialize
from imitation.rewards.serialize import load_reward
from imitation.scripts.config.expert_demos import expert_demos_ex
from imitation.util import logger, util  # need only util, load in this order o/w error


class Planner:
    def __init__(
        self,
        env_name,
        log_dir,
        seed=-1,
        radius=8,
        nsamples_per_ag=50,
        alpha=0.1,
        w_prev=0.0,
        max_episode_steps=60,
        main_env=None,
        beta=None,
        nb_simulations=1000,
    ):
        args = SimpleNamespace()
        args.env_name = env_name
        args.log_dir = log_dir
        args.seed = seed
        args.radius = radius
        args.alpha = alpha
        args.w_prev = w_prev
        args.nsamples_per_ag = nsamples_per_ag
        args.num_vec = 1
        args.nb_simulations = nb_simulations  # Number of MCTS simulations
        args.cInit = 1.25
        args.cBase = 1000
        args.max_rollout_steps = 10
        args.max_episode_length = max_episode_steps
        args.execute_length = 20
        args.planning_agent_id = (
            None  # objects 0,1,2. put object 0 on pos near object 1.
        )
        # default goals by env
        if main_env is None:
            args.main_env = gym.make(args.env_name)
            args.main_env.reset()  # in MCTS_agent this is not resetted! TODO only env (to copy main_env and sim_env and set to main_env. can we just set env as well? why reset_history?)
        else:
            args.main_env = main_env
            args.main_env.reset()
        args.env = gym.make(args.env_name)
        args.sim_env = gym.make(args.env_name)
        # args.goals = copy.deepcopy(args.main_env.goals)
        args.goals = None

        self.agent = MCTS_agent(args)
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.beta = beta

        # pdb.set_trace()

    def set_target_pos(self, planning_agent_id, pos):
        self.args.planning_agent_id = planning_agent_id
        self.args.goals = [[None] for _ in range(self.args.main_env.num_agents)]
        self.args.goals[planning_agent_id] = ["LMA", planning_agent_id, pos, 1]
        self.args.main_env.goals = copy.deepcopy(self.args.goals)
        self.args.env.goals = copy.deepcopy(self.args.goals)
        self.args.sim_env.goals = copy.deepcopy(self.args.goals)
        self.args.planning_agent_id = planning_agent_id

        self.agent.planning_agent_id = planning_agent_id
        self.agent.goals = copy.deepcopy(self.args.goals)
        self.agent.main_env.goals = copy.deepcopy(self.args.goals)
        self.agent.env.goals = copy.deepcopy(self.args.goals)
        self.agent.sim_env.goals = copy.deepcopy(self.args.goals)
        self.agent.args = self.args

    def set_env(self, env):
        # self.args.main_env = env
        pass

    def _get_dist(self, pos1, pos2):
        """get the distance between two 2D positions"""
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

    def are_obs_close(self, obs1, obs2):
        thres = 0.8
        num_entities = self.agent.main_env.num_entities
        len_state = len(obs1) // num_entities
        accum_pos_diff = 0
        for ent in range(num_entities):
            accum_pos_diff += self._get_dist(
                obs1[ent * len_state : (ent * len_state) + 2],
                obs2[ent * len_state : (ent * len_state) + 2],
            )
        print("are_obs_close", accum_pos_diff, thres)
        return accum_pos_diff < thres

    """important that same scheme as in expert_demos fake goal sampling (have gt) for AIRL discrim loss"""

    def select_next_state(
        self,
        value_func,
        venv_norm_obs,
        total_timesteps,
        gt_subgoal=False,
        gt_reward=False,
        G=None,
        aux_value_func=None,
        history=None,
        is_value=True,
        restrict=None,
        stochastic=True,
        is_eval=False,
        ignore_done=False,
    ):
        # print('select_next_state', self.args.main_env.get_full_state())
        n_entities = self.args.main_env.num_agents
        len_state = self.args.main_env.state_dim
        prev_selected_goal = None
        curr_selected_goal = None
        min_x, min_y, max_x, max_y = self.args.main_env.get_boundaries()

        demos = []

        obs = [self.agent.main_env.get_full_state()]
        subgoals = []
        acts_idx = []
        sampled_subgoals = []
        rews = []

        agent_list = (
            list(range(self.agent.main_env.num_entities))
            if restrict is None
            else restrict
        )  # restrict should be a list

        for t in range(total_timesteps):
            # sample object to move
            next_state_list = []
            sampled_ag_id = []
            sampled_acts = []
            for agent_id in agent_list:
                for action in self.agent.main_env._action_space:
                    next_state = self.agent.main_env.transition(
                        agent_id,
                        self.agent.main_env.get_state(),
                        None,
                        action,
                        None,
                    )
                    next_state_list.append(self.agent.main_env.get_full_state_tmp())
                    sampled_acts.append(action)
                    sampled_ag_id.append(agent_id)

            sampled_subgoals_th = torch.as_tensor(
                np.array(next_state_list), device=self.device
            )
            if G is not None:
                # pdb.set_trace()
                B = sampled_subgoals_th.shape[0]
                G = torch.as_tensor(G, device=self.device).float()  # B x N x N x C
                sampled_subgoals_th = sampled_subgoals_th.reshape(
                    (B, n_entities, len_state)
                ).float()  # B x N x D
                goal_values = value_func(sampled_subgoals_th, G)
                value_next_states = goal_values.detach().cpu().numpy()
            else:
                value_next_states, _ = value_func(sampled_subgoals_th)

            # sampling a next state or argmax next state depending on "stochastic"
            value_next_states = np.array(value_next_states)
            next_states_probs_raw = np.exp(self.beta * value_next_states)
            Z = np.sum(np.exp(self.beta * value_next_states))
            next_states_probs = next_states_probs_raw / Z
            # pdb.set_trace()
            if stochastic:
                # next_state and action idx
                selected_idx = np.random.choice(
                    len(next_states_probs), p=next_states_probs
                )
            else:
                selected_idx = np.argmax(next_states_probs)

            selected_actions = [None for _ in range(n_entities)]
            selected_actions[sampled_ag_id[selected_idx]] = sampled_acts[selected_idx]
            curr_obs, rew, done, _ = self.agent.main_env.step(
                selected_actions, planning_agent_id=sampled_ag_id[selected_idx]
            )
            obs.append(curr_obs)
            acts_idx.append(selected_idx % 9)
            sampled_subgoals.append(next_state_list)
            subgoals.append(selected_idx)
            rews.append(rew)
            if done and not ignore_done:
                curr_obs = self.agent.main_env.reset()

                demo = types.TrajectoryWithRew(
                    obs=np.asarray(obs),  # format: (s_0,a_1,s_1,a_2,s_2,...,a_n,s_n)
                    acts=np.asarray(acts_idx),  # dummy values
                    infos=[
                        {
                            "goal": sg,
                            "acts_idx": act_id,  # act_idx and subgoal idx the same per state
                            "sampled_subgoals": next_state_list,
                        }
                        for sg, act_id, next_state_list in zip(
                            subgoals, acts_idx, sampled_subgoals
                        )
                    ],
                    rews=np.asarray(rews),
                )
                obs = [curr_obs]
                subgoals = []
                acts_idx = []
                sampled_subgoals = []
                rews = []
                demos.append(demo)
        if len(acts_idx) > 0:
            demo = types.TrajectoryWithRew(
                obs=np.asarray(obs),  # format: (s_0,a_1,s_1,a_2,s_2,...,a_n,s_n)
                acts=np.asarray(acts_idx),  # dummy values
                infos=[
                    {
                        "goal": sg,
                        "acts_idx": act_id,
                        "sampled_subgoals": next_state_list,
                    }
                    for sg, act_id, next_state_list in zip(
                        subgoals, acts_idx, sampled_subgoals
                    )
                ],
                rews=np.asarray(rews),
            )
            demos.append(demo)
        return demos

    def select_subgoal_and_plan(
        self,
        value_func,
        venv_norm_obs,
        total_timesteps,
        gt_subgoal=False,
        gt_reward=False,
        G=None,
        aux_value_func=None,
        history=None,
        is_value=True,
        restrict=None,
        stochastic=True,
        is_eval=False,
    ):
        """value_func: the value function or reward function learned in AIRL"""
        """agent_states and st_states should be raw states"""
        n_entities = self.args.main_env.num_agents
        prev_selected_goal = None
        curr_selected_goal = None
        min_x, min_y, max_x, max_y = self.args.main_env.get_boundaries()

        demos = []
        t = 0
        while (
            t < total_timesteps
        ):  # per adversarial call, the bigger total_timesteps is handled in adversarial
            self.agent.max_episode_length = min(
                self.agent.max_episode_length, total_timesteps - t
            )
            # print("self.agent.main_env.terminal()", self.agent.main_env.terminal())
            if (
                self.args.goals is None or self.agent.main_env.terminal()
            ) and not gt_subgoal:
                # sample new subgoal
                st_states = self.agent.main_env.agent_states  # start from last pos
                st_obs = self.agent.main_env.get_full_state()
                len_state = len(st_obs) // n_entities
                sampled_pos, sampled_object_id, sampled_subgoal_states = [], [], []
                agent_list = (
                    list(range(self.agent.main_env.num_entities))
                    if restrict is None
                    else restrict
                )  # restrict should be a list
                # for agent_id in agent_list:  # TODO
                nsamples_per_arg_cntr = 0
                # for _ in range(self.args.nsamples_per_ag):
                cnt_attempts = 0
                while nsamples_per_arg_cntr < self.args.nsamples_per_ag:
                    agent_id = random.choice(agent_list)
                    cnt_attempts += 1
                    if cnt_attempts > 10000:  # force it to break after too many samples
                        break
                    x = st_states[agent_id]["pos"][0] + random.choice([-1, 1]) * (
                        round(np.random.uniform(1, self.args.radius), 2)
                    )
                    y = st_states[agent_id]["pos"][1] + random.choice([-1, 1]) * (
                        round(np.random.uniform(1, self.args.radius), 2)
                    )
                    if (
                        x > max_x or x < min_x or y > max_y or y < min_y
                    ):  # check feasibility of subgoals
                        continue
                    sampled_pos.append((x, y))
                    sampled_object_id.append(agent_id)
                    goal_state = st_obs.copy()
                    # change only 1 pos
                    goal_state[agent_id * len_state] = x
                    goal_state[(agent_id * len_state) + 1] = y
                    sampled_subgoal_states.append(goal_state)  # state format
                    nsamples_per_arg_cntr += 1
                self.sampled_subgoal_states = sampled_subgoal_states
                reward_list = []
                if not gt_reward:
                    # eval all these normalized states in value_func
                    if venv_norm_obs is not None:
                        norm_sampled_subgoal_states = venv_norm_obs.normalize_obs(
                            sampled_subgoal_states
                        )
                    else:
                        if type(sampled_subgoal_states) == list:
                            norm_sampled_subgoal_states = np.array(
                                sampled_subgoal_states
                            )
                        else:
                            norm_sampled_subgoal_states = sampled_subgoal_states
                    if is_value:
                        sampled_subgoals_th = torch.as_tensor(
                            norm_sampled_subgoal_states, device=self.device
                        )
                        if G is not None:  # GNN reward TODO
                            # pdb.set_trace()
                            B = sampled_subgoals_th.shape[0]
                            G = torch.as_tensor(
                                G, device=self.device
                            ).float()  # B x N x N x C
                            sampled_subgoals_th = sampled_subgoals_th.reshape(
                                (B, n_entities, len_state)
                            ).float()  # B x N x D
                            goal_values = value_func(sampled_subgoals_th, G)
                            goal_values = goal_values.detach().cpu().numpy()
                        else:
                            goal_values, _ = value_func(sampled_subgoals_th)
                    else:
                        num_nodes = G.shape[0]
                        goal_values_origin = value_func(
                            obs=norm_sampled_subgoal_states,
                            num_nodes=num_nodes,
                            G=G,
                            norm_reward=False,
                        )
                        goal_values_origin = goal_values_origin.detach().cpu().numpy()
                        reward_list = [
                            [goal_values_origin[sampled_goal_id]]
                            for sampled_goal_id in range(len(sampled_subgoal_states))
                        ]
                        if aux_value_func is None and history is None:
                            goal_values = goal_values_origin
                        else:
                            goal_values = copy.deepcopy(goal_values_origin)
                            # goal_values = goal_values_origin.detach().cpu().numpy()
                            # print(goal_values.shape)
                            # pdb.set_trace()
                            if history is not None:
                                for i, sampled_subggoal_state in enumerate(
                                    sampled_subgoal_states
                                ):
                                    sampled_goal_state = [
                                        np.array(
                                            sampled_subggoal_state[
                                                entity_id
                                                * len_state : (
                                                    (entity_id * len_state) + 2
                                                )
                                            ]
                                        )
                                        for entity_id in range(
                                            self.agent.main_env.num_entities
                                        )
                                    ]
                                    min_diff = 1e6
                                    for past_goal_state in history:
                                        curr_diff = 0
                                        # print(
                                        #     "dist",
                                        #     past_goal_state,
                                        #     sampled_goal_state,
                                        # )
                                        for (
                                            post_entity_state,
                                            sampled_entity_state,
                                        ) in zip(past_goal_state, sampled_goal_state):
                                            curr_diff += np.linalg.norm(
                                                post_entity_state - sampled_entity_state
                                            )
                                        min_diff = min(min_diff, curr_diff)
                                    goal_values[i] += self.args.alpha * min_diff
                                    reward_list[i].append(min_diff)
                            if aux_value_func is not None:
                                goal_values_aux_th = aux_value_func[0](
                                    obs=norm_sampled_subgoal_states,
                                    num_nodes=num_nodes,
                                    G=aux_value_func[1],
                                    norm_reward=False,
                                )
                                goal_values_aux = (
                                    goal_values_aux_th.detach().cpu().numpy()
                                )
                                goal_values = (
                                    goal_values - self.args.w_prev * goal_values_aux
                                )
                                for sampled_goal_id in range(
                                    len(sampled_subgoal_states)
                                ):
                                    reward_list[sampled_goal_id].append(
                                        goal_values_aux[sampled_goal_id]
                                    )

                else:  # gt_reward
                    goal_values = []
                    for agent_id, goal_state in zip(
                        sampled_object_id, sampled_subgoal_states
                    ):
                        # convert goal_state to _get_state
                        goal_state_formatted = [
                            self.agent.main_env.convert_obs2state(
                                goal_state[i * len_state : (i + 1) * len_state]
                            )
                            for i in range(n_entities)
                        ]
                        val = self.agent.main_env.get_done_reward(goal_state_formatted)
                        goal_values.append(val)

                for sampled_goal_id in range(len(goal_values)):
                    goal_state = sampled_subgoal_states[sampled_goal_id]
                    goal_state_formatted = [
                        self.agent.main_env.convert_obs2state(
                            goal_state[i * len_state : (i + 1) * len_state]
                        )
                        for i in range(n_entities)
                    ]
                    # print(
                    #     "sampled goal: ",
                    #     [
                    #         sampled_subgoal_states[sampled_goal_id][
                    #             i * len_state : (i * len_state) + 2
                    #         ]
                    #         for i in range(n_entities)
                    #     ],
                    #     "value:",
                    #     goal_values[sampled_goal_id]
                    #     if len(reward_list) == 0
                    #     else (
                    #         reward_list[sampled_goal_id],
                    #         goal_values[sampled_goal_id],
                    #     ),
                    #     "gt value: ",
                    #     self.agent.main_env.get_done_reward(goal_state_formatted),
                    # )

                goal_values = np.array(goal_values)
                subgoal_probs_raw = np.exp(self.beta * goal_values)
                Z = np.sum(np.exp(self.beta * goal_values))
                subgoal_probs = subgoal_probs_raw / Z
                if stochastic:
                    selected_idx = np.random.choice(len(subgoal_probs), p=subgoal_probs)
                else:
                    selected_idx = np.argmax(goal_values)

                if selected_idx is None:
                    return None

                if curr_selected_goal is not None:
                    prev_selected_goal = curr_selected_goal.copy()
                curr_selected_goal = sampled_subgoal_states[selected_idx]

                if (
                    prev_selected_goal is not None
                    and is_eval
                    and (prev_selected_goal is not None)
                    and self.are_obs_close(prev_selected_goal, curr_selected_goal)
                ):
                    self.agent.main_env.reset()
                    print("resetting env, selecting new subgoal")
                    continue

                self.selected_idx = selected_idx
                selected_goal = [
                    "LMA",
                    sampled_object_id[selected_idx],
                    sampled_pos[selected_idx],
                    1,
                ]
                self.set_target_pos(selected_goal[1], selected_goal[2])

                # print("selected idx:", selected_idx)
                # print("set goal", selected_goal)
                # if len(reward_list) > 0:
                #     print(
                #         "goal value:",
                #         reward_list[selected_idx],
                #         goal_values[selected_idx],
                #     )
                # else:
                #     print("goal value:", goal_values[selected_idx])

                # print("G:", G)
                # pdb.set_trace()
                self.agent.main_env.running = True
                self.agent.env.running = True

            demo = self.agent.plan(
                nb_episodes=1,
                sampled_subgoals=self.sampled_subgoal_states,
                selected_subgoal_idx=self.selected_idx,
                save_demos=False,
                max_T=None,
            )
            demos += demo
            t += len(demo[0].acts)

            self.args.goals = None

        return demos

    def select_subgoal(
        self,
        value_func,
        venv_norm_obs,
        total_timesteps,
        gt_subgoal=False,
        gt_reward=False,
        goal=None,
        aux_value_func=None,
        history=None,
        is_value=True,
        restrict=None,
        stochastic=True,
        is_eval=False,
    ):
        """value_func: the value function or reward function learned in AIRL"""
        """agent_states and st_states should be raw states"""
        n_entities = self.args.main_env.num_agents
        prev_selected_goal = None
        curr_selected_goal = None
        min_x, min_y, max_x, max_y = self.args.main_env.get_boundaries()

        G = goal.graph
        edges_transform = goal.edges_transform

        demos = []
        t = 0
        while (
            t < total_timesteps
        ):  # per adversarial call, the bigger total_timesteps is handled in adversarial
            self.agent.max_episode_length = min(
                self.agent.max_episode_length, total_timesteps - t
            )
            # print("self.agent.main_env.terminal()", self.agent.main_env.terminal())
            if (
                self.args.goals is None or self.agent.main_env.terminal()
            ) and not gt_subgoal:
                # sample new subgoal
                st_states = self.agent.main_env.agent_states  # start from last pos
                st_obs = self.agent.main_env.get_full_state()
                len_state = len(st_obs) // n_entities
                sampled_pos, sampled_object_id, sampled_subgoal_states = [], [], []

                agent_list = (
                    list(range(self.agent.main_env.num_entities))
                    if restrict is None
                    else restrict
                )  # restrict should be a list
                # for agent_id in agent_list:  # TODO
                nsamples_per_arg_cntr = 0
                # for _ in range(self.args.nsamples_per_ag):
                cnt_attempts = 0
                while nsamples_per_arg_cntr < self.args.nsamples_per_ag:

                    goal_state = st_obs.copy()

                    cnt_attempts += 1
                    if (
                        cnt_attempts > 100000000
                    ):  # force it to break after too many samples
                        break
                    feasible = True
                    for agent_id in agent_list:
                        x = st_states[agent_id]["pos"][0] + np.random.uniform(
                            -self.args.radius, self.args.radius
                        )
                        y = st_states[agent_id]["pos"][1] + np.random.uniform(
                            -self.args.radius, self.args.radius
                        )
                        if (
                            x > max_x or x < min_x or y > max_y or y < min_y
                        ):  # check feasibility of subgoals
                            feasible = False
                            break
                        for other_agent_id in range(self.agent.main_env.num_entities):
                            if other_agent_id != agent_id:
                                if (
                                    (x - goal_state[other_agent_id * len_state]) ** 2
                                    + (y - goal_state[other_agent_id * len_state + 1])
                                    ** 2
                                ) ** 0.5 < 1.6:
                                    feasible = False
                                    break
                        if not feasible:
                            break
                        goal_state[agent_id * len_state] = x
                        goal_state[(agent_id * len_state) + 1] = y
                    if not feasible:
                        continue
                    sampled_subgoal_states.append(goal_state)  # state format
                    nsamples_per_arg_cntr += 1

                    sampled_pos.append((x, y))  # TODO remove this
                    sampled_object_id.append(agent_id)

                self.sampled_subgoal_states = sampled_subgoal_states
                # pdb.set_trace() #check sampled_subgoal_states
                reward_list = []
                if not gt_reward:
                    # eval all these normalized states in value_func
                    if venv_norm_obs is not None:
                        norm_sampled_subgoal_states = venv_norm_obs.normalize_obs(
                            sampled_subgoal_states
                        )
                    else:
                        if type(sampled_subgoal_states) == list:
                            norm_sampled_subgoal_states = np.array(
                                sampled_subgoal_states
                            )
                        else:
                            norm_sampled_subgoal_states = sampled_subgoal_states
                    if is_value:
                        sampled_subgoals_th = torch.as_tensor(
                            norm_sampled_subgoal_states, device=self.device
                        )
                        if G is not None:  # GNN reward TODO
                            # pdb.set_trace()
                            B = sampled_subgoals_th.shape[0]
                            G = torch.as_tensor(
                                G, device=self.device
                            ).float()  # B x N x N x C
                            sampled_subgoals_th = sampled_subgoals_th.reshape(
                                (B, n_entities, len_state)
                            ).float()  # B x N x D
                            goal_values = value_func(sampled_subgoals_th, G)
                            goal_values = goal_values.detach().cpu().numpy()
                        else:
                            goal_values, _ = value_func(sampled_subgoals_th)
                    else:
                        num_nodes = G.shape[0]
                        goal_values_origin = value_func(
                            obs=norm_sampled_subgoal_states,
                            num_nodes=num_nodes,
                            G=G,
                            norm_reward=False,
                        )
                        goal_values_origin = goal_values_origin.detach().cpu().numpy()
                        reward_list = [
                            [goal_values_origin[sampled_goal_id]]
                            for sampled_goal_id in range(len(sampled_subgoal_states))
                        ]
                        if aux_value_func is None and history is None:
                            goal_values = goal_values_origin
                        else:
                            goal_values = copy.deepcopy(goal_values_origin)
                            # goal_values = goal_values_origin.detach().cpu().numpy()
                            # print(goal_values.shape)
                            # pdb.set_trace()
                            if history is not None:
                                for i, sampled_subggoal_state in enumerate(
                                    sampled_subgoal_states
                                ):
                                    sampled_goal_state = [
                                        np.array(
                                            sampled_subggoal_state[
                                                entity_id
                                                * len_state : (
                                                    (entity_id * len_state) + 2
                                                )
                                            ]
                                        )
                                        for entity_id in range(
                                            self.agent.main_env.num_entities
                                        )
                                    ]
                                    min_diff = 1e6
                                    for past_goal_state in history:
                                        curr_diff = 0
                                        # print(
                                        #     "dist",
                                        #     past_goal_state,
                                        #     sampled_goal_state,
                                        # )
                                        for (
                                            post_entity_state,
                                            sampled_entity_state,
                                        ) in zip(past_goal_state, sampled_goal_state):
                                            curr_diff += np.linalg.norm(
                                                post_entity_state - sampled_entity_state
                                            )
                                        min_diff = min(min_diff, curr_diff)
                                    goal_values[i] += self.args.alpha * min_diff
                                    reward_list[i].append(min_diff)
                            if aux_value_func is not None:
                                prev_state = np.array([st_obs])
                                prev_values_aux_th = aux_value_func[0](
                                    obs=prev_state,
                                    num_nodes=num_nodes,
                                    G=aux_value_func[1],
                                    norm_reward=False,
                                )
                                prev_value = (
                                    prev_values_aux_th.detach().cpu().numpy()[0]
                                )

                                goal_values_aux_th = aux_value_func[0](
                                    obs=norm_sampled_subgoal_states,
                                    num_nodes=num_nodes,
                                    G=aux_value_func[1],
                                    norm_reward=False,
                                )
                                goal_values_aux = (
                                    goal_values_aux_th.detach().cpu().numpy()
                                )
                                goal_values = (
                                    goal_values - self.args.w_prev * goal_values_aux
                                )

                                # goal_values -= 0.2 * (goal_values_aux - prev_value)

                                for sampled_goal_id in range(
                                    len(sampled_subgoal_states)
                                ):
                                    reward_list[sampled_goal_id].append(
                                        goal_values_aux[sampled_goal_id]
                                    )
                                    reward_list[sampled_goal_id].append(
                                        goal_values_aux[sampled_goal_id] - prev_value
                                    )

                else:  # gt_reward
                    goal_values = []
                    for agent_id, goal_state in zip(
                        sampled_object_id, sampled_subgoal_states
                    ):
                        # convert goal_state to _get_state
                        goal_state_formatted = [
                            self.agent.main_env.convert_obs2state(
                                goal_state[i * len_state : (i + 1) * len_state]
                            )
                            for i in range(n_entities)
                        ]

                        val = self.agent.main_env.get_done_reward(goal_state_formatted)
                        goal_values.append(val)

                for sampled_goal_id in range(len(goal_values)):
                    goal_state = sampled_subgoal_states[sampled_goal_id]
                    goal_state_formatted = [
                        self.agent.main_env.convert_obs2state(
                            goal_state[i * len_state : (i + 1) * len_state]
                        )
                        for i in range(n_entities)
                    ]

                goal_values = np.array(goal_values)
                subgoal_probs_raw = np.exp(self.beta * goal_values)
                Z = np.sum(np.exp(self.beta * goal_values))
                subgoal_probs = subgoal_probs_raw / Z
                if stochastic:
                    selected_idx = np.random.choice(len(subgoal_probs), p=subgoal_probs)
                else:
                    selected_idx = np.argmax(goal_values)

                if selected_idx is None:
                    return None

                if curr_selected_goal is not None:
                    prev_selected_goal = curr_selected_goal.copy()
                curr_selected_goal = sampled_subgoal_states[selected_idx]

                self.selected_idx = selected_idx
                selected_goal = [
                    "LMA",
                    sampled_object_id[selected_idx],
                    sampled_pos[selected_idx],
                    1,
                ]
                self.set_target_pos(selected_goal[1], selected_goal[2])

                # print("selected idx:", selected_idx)
                # print(
                #     "selected subgoal:",
                #     [
                #         self.sampled_subgoal_states[self.selected_idx][
                #             self.agent.main_env.state_dim
                #             * entity_id : self.agent.main_env.state_dim
                #             * entity_id
                #             + 2
                #         ]
                #         for entity_id in range(self.agent.main_env.num_entities)
                #     ],
                # )
                # if len(reward_list) > 0:
                #     print(
                #         "goal value:",
                #         reward_list[selected_idx],
                #         goal_values[selected_idx],
                #     )
                # else:
                #     print("goal value:", goal_values[selected_idx])

                # print("G:", G)
                # pdb.set_trace()
                self.agent.main_env.running = True
                self.agent.env.running = True

            subgoal_agent_states = self.agent.main_env.convert_fullobs2state(
                self.sampled_subgoal_states[self.selected_idx]
            )
            self.agent.main_env.set_state(subgoal_agent_states)
            demo = self.agent.main_env.get_full_state()

            demos.append(demo)
            t += 1

            self.args.goals = None

        return [[demos[-1]]]

    def select_subgoal_eval_cost(
        self,
        value_func,
        venv_norm_obs,
        total_timesteps,
        gt_subgoal=False,
        gt_reward=False,
        goal=None,
        aux_value_func=None,
        history=None,
        is_value=True,
        restrict=None,
        stochastic=True,
        is_eval=False,
    ):
        """value_func: the value function or reward function learned in AIRL"""
        """agent_states and st_states should be raw states"""
        n_entities = self.args.main_env.num_agents
        prev_selected_goal = None
        curr_selected_goal = None
        min_x, min_y, max_x, max_y = self.args.main_env.get_boundaries()

        G = goal.graph
        edges_transform = goal.edges_transform

        demos = []
        rewards = []

        t = 0
        while t < total_timesteps:
            self.agent.max_episode_length = min(
                self.agent.max_episode_length, total_timesteps - t
            )
            # print("self.agent.main_env.terminal()", self.agent.main_env.terminal())
            if (
                self.args.goals is None or self.agent.main_env.terminal()
            ) and not gt_subgoal:
                # sample new subgoal
                st_states = self.agent.main_env.agent_states  # start from last pos
                st_obs = self.agent.main_env.get_full_state()
                len_state = len(st_obs) // n_entities
                (
                    sampled_pos,
                    sampled_object_id,
                    sampled_subgoal_states,
                    sampled_subgoal_states_dist,
                ) = ([], [], [], [])

                agent_list = (
                    list(range(self.agent.main_env.num_entities))
                    if restrict is None
                    else restrict
                )  # restrict should be a list
                # for agent_id in agent_list:  # TODO
                nsamples_per_arg_cntr = 0
                # for _ in range(self.args.nsamples_per_ag):
                cnt_attempts = 0
                while nsamples_per_arg_cntr < self.args.nsamples_per_ag:

                    goal_state = st_obs.copy()

                    cnt_attempts += 1
                    if (
                        cnt_attempts > 100000000
                    ):  # force it to break after too many samples
                        break

                    feasible = True
                    dist = 0
                    for agent_id in agent_list:
                        dx = np.random.uniform(-self.args.radius, self.args.radius)
                        x = st_states[agent_id]["pos"][0] + dx
                        dy = np.random.uniform(-self.args.radius, self.args.radius)
                        y = st_states[agent_id]["pos"][1] + dy
                        if (
                            x > max_x or x < min_x or y > max_y or y < min_y
                        ):  # check feasibility of subgoals
                            feasible = False
                            break
                        for other_agent_id in range(self.agent.main_env.num_entities):
                            if other_agent_id != agent_id:
                                if (
                                    (x - goal_state[other_agent_id * len_state]) ** 2
                                    + (y - goal_state[other_agent_id * len_state + 1])
                                    ** 2
                                ) ** 0.5 < 1.6:
                                    feasible = False
                                    break
                        if not feasible:
                            break
                        goal_state[agent_id * len_state] = x
                        goal_state[(agent_id * len_state) + 1] = y
                        dist += (dx**2 + dy**2) ** 0.5
                    if not feasible:
                        continue

                    sampled_subgoal_states_dist.append(dist)

                    sampled_subgoal_states.append(goal_state)
                    nsamples_per_arg_cntr += 1

                    sampled_pos.append((x, y))
                    sampled_object_id.append(agent_id)

                self.sampled_subgoal_states = sampled_subgoal_states
                reward_list = []
                if not gt_reward:
                    # eval all these normalized states in value_func
                    if venv_norm_obs is not None:
                        norm_sampled_subgoal_states = venv_norm_obs.normalize_obs(
                            sampled_subgoal_states
                        )
                    else:
                        if type(sampled_subgoal_states) == list:
                            norm_sampled_subgoal_states = np.array(
                                sampled_subgoal_states
                            )
                        else:
                            norm_sampled_subgoal_states = sampled_subgoal_states
                    if is_value:
                        sampled_subgoals_th = torch.as_tensor(
                            norm_sampled_subgoal_states, device=self.device
                        )
                        if G is not None:  # GNN reward TODO
                            # pdb.set_trace()
                            B = sampled_subgoals_th.shape[0]
                            G = torch.as_tensor(
                                G, device=self.device
                            ).float()  # B x N x N x C
                            sampled_subgoals_th = sampled_subgoals_th.reshape(
                                (B, n_entities, len_state)
                            ).float()  # B x N x D
                            goal_values = value_func(sampled_subgoals_th, G)
                            goal_values = goal_values.detach().cpu().numpy()
                        else:
                            goal_values, _ = value_func(sampled_subgoals_th)
                    else:
                        num_nodes = G.shape[0]
                        goal_values_origin = value_func(
                            obs=norm_sampled_subgoal_states,
                            num_nodes=num_nodes,
                            G=G,
                            norm_reward=False,
                        )
                        goal_values_origin = goal_values_origin.detach().cpu().numpy()
                        reward_list = [
                            [goal_values_origin[sampled_goal_id]]
                            for sampled_goal_id in range(len(sampled_subgoal_states))
                        ]
                        if aux_value_func is None and history is None:
                            goal_values = goal_values_origin
                        else:
                            goal_values = copy.deepcopy(goal_values_origin)
                            if history is not None:
                                for i, sampled_subggoal_state in enumerate(
                                    sampled_subgoal_states
                                ):
                                    sampled_goal_state = [
                                        np.array(
                                            sampled_subggoal_state[
                                                entity_id
                                                * len_state : (
                                                    (entity_id * len_state) + 2
                                                )
                                            ]
                                        )
                                        for entity_id in range(
                                            self.agent.main_env.num_entities
                                        )
                                    ]
                                    min_diff = 1e6
                                    for past_goal_state in history:
                                        curr_diff = 0
                                        for (
                                            post_entity_state,
                                            sampled_entity_state,
                                        ) in zip(past_goal_state, sampled_goal_state):
                                            curr_diff += np.linalg.norm(
                                                post_entity_state - sampled_entity_state
                                            )
                                        min_diff = min(min_diff, curr_diff)
                                    goal_values[i] += self.args.alpha * min_diff
                                    reward_list[i].append(min_diff)
                            if aux_value_func is not None:
                                prev_state = np.array([st_obs])
                                prev_values_aux_th = aux_value_func[0](
                                    obs=prev_state,
                                    num_nodes=num_nodes,
                                    G=aux_value_func[1],
                                    norm_reward=False,
                                )
                                prev_value = (
                                    prev_values_aux_th.detach().cpu().numpy()[0]
                                )

                                goal_values_aux_th = aux_value_func[0](
                                    obs=norm_sampled_subgoal_states,
                                    num_nodes=num_nodes,
                                    G=aux_value_func[1],
                                    norm_reward=False,
                                )
                                goal_values_aux = (
                                    goal_values_aux_th.detach().cpu().numpy()
                                )
                                goal_values = (
                                    goal_values - self.args.w_prev * goal_values_aux
                                )

                                # goal_values -= 0.2 * (goal_values_aux - prev_value)

                                for sampled_goal_id in range(
                                    len(sampled_subgoal_states)
                                ):
                                    reward_list[sampled_goal_id].append(
                                        goal_values_aux[sampled_goal_id]
                                    )
                                    reward_list[sampled_goal_id].append(
                                        goal_values_aux[sampled_goal_id] - prev_value
                                    )

                else:  # gt_reward
                    goal_values = []
                    for agent_id, goal_state in zip(
                        sampled_object_id, sampled_subgoal_states
                    ):
                        # convert goal_state to _get_state
                        goal_state_formatted = [
                            self.agent.main_env.convert_obs2state(
                                goal_state[i * len_state : (i + 1) * len_state]
                            )
                            for i in range(n_entities)
                        ]
                        val = self.agent.main_env.get_done_reward(goal_state_formatted)
                        goal_values.append(val)

                for sampled_goal_id in range(len(goal_values)):
                    goal_state = sampled_subgoal_states[sampled_goal_id]
                    goal_state_formatted = [
                        self.agent.main_env.convert_obs2state(
                            goal_state[i * len_state : (i + 1) * len_state]
                        )
                        for i in range(n_entities)
                    ]

                goal_values = np.array(goal_values)
                goal_dists = np.array(sampled_subgoal_states_dist)

                # print(
                #     "stats of dist:",
                #     np.min(goal_dists),
                #     np.max(goal_dists),
                #     np.mean(goal_dists),
                # )

                goal_values = goal_values - goal_dists * 0.02

                selected_idx = np.argmax(goal_values)

                if selected_idx is None:
                    return None

                if curr_selected_goal is not None:
                    prev_selected_goal = curr_selected_goal.copy()
                curr_selected_goal = sampled_subgoal_states[selected_idx]

                self.selected_idx = selected_idx
                selected_goal = [
                    "LMA",
                    sampled_object_id[selected_idx],
                    sampled_pos[selected_idx],
                    1,
                ]
                self.set_target_pos(selected_goal[1], selected_goal[2])

                # print("selected idx:", selected_idx)
                # # print("set goal", selected_goal)
                # print(
                #     "selected subgoal:",
                #     [
                #         self.sampled_subgoal_states[self.selected_idx][
                #             self.agent.main_env.state_dim
                #             * entity_id : self.agent.main_env.state_dim
                #             * entity_id
                #             + 2
                #         ]
                #         for entity_id in range(self.agent.main_env.num_entities)
                #     ],
                # )
                # if len(reward_list) > 0:
                #     print(
                #         "goal value:",
                #         reward_list[selected_idx],
                #         goal_values[selected_idx],
                #     )
                # else:
                #     print("goal value:", goal_values[selected_idx])
                rewards.append(goal_values[selected_idx])

                # print("G:", G)
                # pdb.set_trace()
                self.agent.main_env.running = True
                self.agent.env.running = True

            subgoal_agent_states = self.agent.main_env.convert_fullobs2state(
                self.sampled_subgoal_states[self.selected_idx]
            )
            self.agent.main_env.set_state(subgoal_agent_states)
            demo = self.agent.main_env.get_full_state()

            demos.append(demo)
            t += 1

            self.args.goals = None
        return [[demos[-1]]], [[rewards[-1]]]

    def select_subgoal_random(
        self,
        value_func,
        venv_norm_obs,
        total_timesteps,
        gt_subgoal=False,
        gt_reward=False,
        goal=None,
        aux_value_func=None,
        history=None,
        is_value=True,
        restrict=None,
        stochastic=True,
        is_eval=False,
    ):
        """value_func: the value function or reward function learned in AIRL"""
        """agent_states and st_states should be raw states"""
        n_entities = self.args.main_env.num_agents
        prev_selected_goal = None
        curr_selected_goal = None
        min_x, min_y, max_x, max_y = self.args.main_env.get_boundaries()

        G = goal.graph
        edges_transform = goal.edges_transform

        demos = []
        rewards = []
        t = 0
        while (
            t < total_timesteps
        ):  # per adversarial call, the bigger total_timesteps is handled in adversarial
            self.agent.max_episode_length = min(
                self.agent.max_episode_length, total_timesteps - t
            )
            print("self.agent.main_env.terminal()", self.agent.main_env.terminal())
            if (
                self.args.goals is None or self.agent.main_env.terminal()
            ) and not gt_subgoal:
                # sample new subgoal
                st_states = self.agent.main_env.agent_states  # start from last pos
                st_obs = self.agent.main_env.get_full_state()
                len_state = len(st_obs) // n_entities
                sampled_pos, sampled_object_id, sampled_subgoal_states = [], [], []

                agent_list = (
                    list(range(self.agent.main_env.num_entities))
                    if restrict is None
                    else restrict
                )  # restrict should be a list
                # for agent_id in agent_list:  # TODO
                nsamples_per_arg_cntr = 0
                # for _ in range(self.args.nsamples_per_ag):
                cnt_attempts = 0
                while nsamples_per_arg_cntr < self.args.nsamples_per_ag:

                    goal_state = st_obs.copy()

                    cnt_attempts += 1
                    if (
                        cnt_attempts > 100000000
                    ):  # force it to break after too many samples
                        break

                    feasible = True
                    for agent_id in agent_list:
                        x = st_states[agent_id]["pos"][0] + np.random.uniform(
                            -self.args.radius, self.args.radius
                        )
                        y = st_states[agent_id]["pos"][1] + np.random.uniform(
                            -self.args.radius, self.args.radius
                        )
                        if (
                            x > max_x or x < min_x or y > max_y or y < min_y
                        ):  # check feasibility of subgoals
                            feasible = False
                            break
                        for other_agent_id in range(self.agent.main_env.num_entities):
                            if other_agent_id != agent_id:
                                if (
                                    (x - goal_state[other_agent_id * len_state]) ** 2
                                    + (y - goal_state[other_agent_id * len_state + 1])
                                    ** 2
                                ) ** 0.5 < 1.6:
                                    feasible = False
                                    break
                        if not feasible:
                            break
                        goal_state[agent_id * len_state] = x
                        goal_state[(agent_id * len_state) + 1] = y
                    if not feasible:
                        continue

                    sampled_subgoal_states.append(goal_state)  # state format
                    nsamples_per_arg_cntr += 1

                    sampled_pos.append((x, y))  # TODO remove this
                    sampled_object_id.append(agent_id)

                self.sampled_subgoal_states = sampled_subgoal_states
                # pdb.set_trace() #check sampled_subgoal_states
                reward_list = []
                if not gt_reward:
                    # eval all these normalized states in value_func
                    if venv_norm_obs is not None:
                        norm_sampled_subgoal_states = venv_norm_obs.normalize_obs(
                            sampled_subgoal_states
                        )
                    else:
                        if type(sampled_subgoal_states) == list:
                            norm_sampled_subgoal_states = np.array(
                                sampled_subgoal_states
                            )
                        else:
                            norm_sampled_subgoal_states = sampled_subgoal_states
                    if is_value:
                        sampled_subgoals_th = torch.as_tensor(
                            norm_sampled_subgoal_states, device=self.device
                        )
                        if G is not None:  # GNN reward TODO
                            # pdb.set_trace()
                            B = sampled_subgoals_th.shape[0]
                            G = torch.as_tensor(
                                G, device=self.device
                            ).float()  # B x N x N x C
                            sampled_subgoals_th = sampled_subgoals_th.reshape(
                                (B, n_entities, len_state)
                            ).float()  # B x N x D
                            goal_values = value_func(sampled_subgoals_th, G)
                            goal_values = goal_values.detach().cpu().numpy()
                        else:
                            goal_values, _ = value_func(sampled_subgoals_th)
                    else:
                        num_nodes = G.shape[0]
                        goal_values_origin = value_func(
                            obs=norm_sampled_subgoal_states,
                            num_nodes=num_nodes,
                            G=G,
                            norm_reward=False,
                        )
                        goal_values_origin = goal_values_origin.detach().cpu().numpy()
                        reward_list = [
                            [goal_values_origin[sampled_goal_id]]
                            for sampled_goal_id in range(len(sampled_subgoal_states))
                        ]
                        if aux_value_func is None and history is None:
                            goal_values = goal_values_origin
                        else:
                            goal_values = copy.deepcopy(goal_values_origin)
                            # goal_values = goal_values_origin.detach().cpu().numpy()
                            # print(goal_values.shape)
                            # pdb.set_trace()
                            if history is not None:
                                for i, sampled_subggoal_state in enumerate(
                                    sampled_subgoal_states
                                ):
                                    sampled_goal_state = [
                                        np.array(
                                            sampled_subggoal_state[
                                                entity_id
                                                * len_state : (
                                                    (entity_id * len_state) + 2
                                                )
                                            ]
                                        )
                                        for entity_id in range(
                                            self.agent.main_env.num_entities
                                        )
                                    ]
                                    min_diff = 1e6
                                    for past_goal_state in history:
                                        curr_diff = 0
                                        # print(
                                        #     "dist",
                                        #     past_goal_state,
                                        #     sampled_goal_state,
                                        # )
                                        for (
                                            post_entity_state,
                                            sampled_entity_state,
                                        ) in zip(past_goal_state, sampled_goal_state):
                                            curr_diff += np.linalg.norm(
                                                post_entity_state - sampled_entity_state
                                            )
                                        min_diff = min(min_diff, curr_diff)
                                    goal_values[i] += self.args.alpha * min_diff
                                    reward_list[i].append(min_diff)
                            if aux_value_func is not None:
                                prev_state = np.array([st_obs])
                                prev_values_aux_th = aux_value_func[0](
                                    obs=prev_state,
                                    num_nodes=num_nodes,
                                    G=aux_value_func[1],
                                    norm_reward=False,
                                )
                                prev_value = (
                                    prev_values_aux_th.detach().cpu().numpy()[0]
                                )

                                goal_values_aux_th = aux_value_func[0](
                                    obs=norm_sampled_subgoal_states,
                                    num_nodes=num_nodes,
                                    G=aux_value_func[1],
                                    norm_reward=False,
                                )
                                goal_values_aux = (
                                    goal_values_aux_th.detach().cpu().numpy()
                                )
                                goal_values = (
                                    goal_values - self.args.w_prev * goal_values_aux
                                )

                                # goal_values -= 0.2 * (goal_values_aux - prev_value)

                                for sampled_goal_id in range(
                                    len(sampled_subgoal_states)
                                ):
                                    reward_list[sampled_goal_id].append(
                                        goal_values_aux[sampled_goal_id]
                                    )
                                    reward_list[sampled_goal_id].append(
                                        goal_values_aux[sampled_goal_id] - prev_value
                                    )

                else:  # gt_reward
                    goal_values = []
                    for agent_id, goal_state in zip(
                        sampled_object_id, sampled_subgoal_states
                    ):
                        # convert goal_state to _get_state
                        goal_state_formatted = [
                            self.agent.main_env.convert_obs2state(
                                goal_state[i * len_state : (i + 1) * len_state]
                            )
                            for i in range(n_entities)
                        ]
                        val = self.agent.main_env.get_done_reward(goal_state_formatted)
                        goal_values.append(val)

                for sampled_goal_id in range(len(goal_values)):
                    goal_state = sampled_subgoal_states[sampled_goal_id]
                    goal_state_formatted = [
                        self.agent.main_env.convert_obs2state(
                            goal_state[i * len_state : (i + 1) * len_state]
                        )
                        for i in range(n_entities)
                    ]

                goal_values = np.array(goal_values)
                subgoal_probs_raw = np.exp(self.beta * goal_values)
                Z = np.sum(np.exp(self.beta * goal_values))
                subgoal_probs = subgoal_probs_raw / Z
                selected_idx = np.random.choice(len(subgoal_probs))

                if selected_idx is None:
                    return None

                if curr_selected_goal is not None:
                    prev_selected_goal = curr_selected_goal.copy()
                curr_selected_goal = sampled_subgoal_states[selected_idx]

                self.selected_idx = selected_idx
                selected_goal = [
                    "LMA",
                    sampled_object_id[selected_idx],
                    sampled_pos[selected_idx],
                    1,
                ]
                self.set_target_pos(selected_goal[1], selected_goal[2])

                print("selected idx:", selected_idx)
                print(
                    "selected subgoal:",
                    [
                        self.sampled_subgoal_states[self.selected_idx][
                            self.agent.main_env.state_dim
                            * entity_id : self.agent.main_env.state_dim
                            * entity_id
                            + 2
                        ]
                        for entity_id in range(self.agent.main_env.num_entities)
                    ],
                )
                if len(reward_list) > 0:
                    print(
                        "goal value:",
                        reward_list[selected_idx],
                        goal_values[selected_idx],
                    )
                else:
                    print("goal value:", goal_values[selected_idx])

                print("G:", G)
                # pdb.set_trace()
                self.agent.main_env.running = True
                self.agent.env.running = True

            subgoal_agent_states = self.agent.main_env.convert_fullobs2state(
                self.sampled_subgoal_states[self.selected_idx]
            )
            self.agent.main_env.set_state(subgoal_agent_states)
            demo = self.agent.main_env.get_full_state()

            demos.append(demo)
            t += 1

            self.args.goals = None

        return [demos]


parser = argparse.ArgumentParser()
parser.add_argument("--env-name", type=str)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--log-dir", type=str)
parser.add_argument("--gt-subgoal", action="store_true")
parser.add_argument("--gt-reward", action="store_true")

if __name__ == "__main__":
    # args = parser.parse_args()
    # args.seed = random.seed(datetime.now())
    planner = Planner(args.env_name, args.log_dir, args.seed)

    # unit test gt subgoal
    if args.gt_subgoal:
        planner.set_target_pos(0, (15, 15))  # gt
        planner.select_subgoal_and_plan(
            value_func=None, venv_norm_obs=None, total_timesteps=1000, gt_subgoal=True
        )

    # unit test gt reward
    if args.gt_reward:
        planner.select_subgoal_and_plan(
            value_func=None, venv_norm_obs=None, total_timesteps=1000, gt_reward=True
        )
