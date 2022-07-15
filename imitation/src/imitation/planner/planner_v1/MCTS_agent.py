from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
from pathlib import Path
import sys
import random
import time
import math
import copy
import pickle
import importlib
import multiprocessing
import os
import pdb

from imitation.planner.planner_v1.MCTS import *

# from utils import *
# from envs.box2d import *

from imitation.data import types

np.set_printoptions(precision=2, suppress=True)


def _str2class(str):
    return getattr(sys.modules[__name__], str)


def select_action(nb_steps, sim_env, mcts, expected_action):
    # sim_env.set_state(agents_pos, agents_vel, nb_steps)
    initState = sim_env.get_state()
    # pdb.set_trace()  #self.main_env.env_method('get_full_obs'), self.main_env.get_attr('running')
    if sim_env.terminal():
        print('simenv state', [(state['pos'],state['angle']) for state in sim_env.get_state()])
        terminal = True
        # res[sample_id] = None
        action = None #'stop'
        children_visit = np.ones_like(sim_env._action_space)        
        selected_child_index = np.where(np.array(sim_env._action_space)=='stop')[0][0]
        children_visit[selected_child_index] = 1000 - len(sim_env._action_space) + 1
        return action, children_visit, selected_child_index
    rootAction = None  # self.sim_envs[agent_id].action_space[-1] #random.choice(self.env.action_space)
    # rootNode = Node(id={rootAction: initState}, #delete
    #                 num_visited=0,
    #                 sum_value=0,
    #                 is_expanded=True)
    rootNode = Node(
        id=0, action=None, state=initState, num_visited=0, sum_value=0, is_expanded=True
    )
    currNode = rootNode
    # currState = list(currNode.id.values())[0]
    next_root, action, children_visit, selected_child_index = mcts.run(
        currNode, nb_steps, expected_action
    )
    return action, children_visit, selected_child_index


class MCTS_agent:
    """
    MCTS for 1 agent
    """

    def __init__(self, args):
        # random.seed(args.seed)
        self.main_env = args.main_env
        self.env = args.env
        self.sim_env = args.sim_env  # TODO needed?
        self.record_dir = args.log_dir
        self.args = args
        self.max_episode_length = 20 #args.max_episode_length  # 100 #for main env
        self.num_agents = self.main_env.num_agents
        self.plan_length = 20 #args.plan_length  # 20 #for env
        self.goals = args.goals
        self.execute_length = args.execute_length
        self.planning_agent_id = args.planning_agent_id

    def goal_decomposition(self, env, agent_id, goal):
        if goal[0] == "LMO" and goal[3] == 1:
            return env.get_subgoals_put(
                agent_id, goal
            )  # TODO in env, adjust to goal location
        elif (goal[0] == "LMA" and goal[3] == 1) or (goal[0] == "TE" and goal[3] == 1):
            return env.get_subgoals_goto(agent_id, goal)
        return goal  # TODO negative goal[3]?

    def _transition(
        self, agent_id, curr_state, action, expected_action, c_init, c_base
    ):
        # PHASE - (self.agent_id, curr_state, curr_certainty, action, expected_action, self.c_init, self.c_base)
        return self.sim_env.transition(
            agent_id,
            curr_state,
            None,
            action,
            expected_action,
            c_init,
            c_base,
            update_certainty=False,
        )

    # def _transition(self):
    #     curr_env = self.sim_env
    #     # MCTS - (self.agent_id, curr_state, action, expected_action, self.c_init, self.c_base)
    #     def simple_transition(agent_id, curr_state, action, expected_action, c_init, c_base):
    #         """transition func (simulate one step)"""
    #         curr_env.env_method('_setup_tmp', curr_env.get_attr('env_id')[0], curr_state)
    #         curr_env.env_method('_send_action_tmp', agent_id, action)
    #         curr_env.env_method('_step_tmp', update_certainty=False)
    #         next_state = curr_env.env_method('_get_state_tmp')[0]
    #         return next_state

    #     return simple_transition

    def _terminal(self, agent_id, curr_state=None, t=None):
        return self.sim_env.terminal(agent_id)

    def _get_reward_state(
        self,
        agent_id,
        curr_state,
        action,
        t=0,
        T=0,
        goal=None,
        prev_certainty=None,
        curr_certainty=None,
    ):
        return self.sim_env.get_reward_state(
            agent_id, curr_state, action, t, T, goal, prev_certainty, curr_certainty
        )

    def rollout(self, record=False, episode_id=None, max_T=None):
        nb_steps = 0
        # self.main_env.reset() #in planner

        final_subgoals = [None] * self.num_agents
        all_modes = []
        while self.main_env.running:
            cur_all_modes = self.plan_select_high_level_goals(max_T=max_T)
            plans, subgoals = cur_all_modes["plans"], cur_all_modes["subgoals"]
            print("rollout plans", plans)
            all_modes.append(cur_all_modes)
            T = len(plans[0])  # min(len(plans[0]), self.execute_length)
            print("T", T)
            for t in range(T):
                if not self.main_env.running:
                    break
                # for agent_id in range(self.num_agents):
                #     if final_subgoals[agent_id] is None:
                #         final_subgoals[agent_id] = [subgoals[agent_id][t]]
                #     else:
                #         final_subgoals[agent_id] += [subgoals[agent_id][t]]
                self.main_env.step(
                    [plans[agent_id][t] for agent_id in range(self.num_agents)],
                    planning_agent_id=self.planning_agent_id,
                )
                print("main_env state:", [state['pos'] for state in self.main_env.get_state()])
                print('self.main_env.running',self.main_env.running)
                print('self.main_env.terminal',self.main_env.terminal())
            #     print("self.main_env.steps",self.main_env.steps)
            #     print("self.max_episode_length",self.max_episode_length)
            #     if self.main_env.steps >= self.max_episode_length:
            #         break
            # print("outer self.main_env.steps",self.main_env.steps)
            # print("outer self.max_episode_length",self.max_episode_length)
            # if self.main_env.steps >= self.max_episode_length:
            #     break
        return all_modes

    def plan_select_high_level_goals(self, max_T=None):
        (
            trajs,
            plans,
            plans_idx,
            subgoals,
            Vs,
            Rs,
            visitations,
        ) = self.plan_given_high_level_goals(max_T=max_T)
        print("Vs", Vs)

        return {
            "trajs": trajs,
            "plans": plans,
            "plans_idx": plans_idx,
            "subgoals": subgoals,
            "Vs": Vs,
            "Rs": Rs,
            "visitation": visitations,
        }

    def plan_given_high_level_goals(self, max_T=None):
        plan_length = (
            min(self.plan_length, max_T) if max_T is not None else self.plan_length
        )
        print("self.main_env.steps", self.main_env.steps)

        # if self.main_env.steps == 0: #TODO
        #     nb_steps = 0
        #     self.env.reset()

        #     for t in range(self.main_env.steps):
        #         for agent_id in range(self.num_agents):
        #             print('main env actions',self.main_env.actions)
        #             self.env.send_action(agent_id, self.main_env.actions[agent_id][t])
        #         # self.env.step(self.main_env.actions[self.planning_agent_id][t],
        #         #                 planning_agent_id=self.planning_agent_id)
        # else:
        #     nb_steps = self.main_env.steps
        #     self.env.running = True #TODO is this good here
        # self.env.reset_history() #TODO why need this?
        # nb_steps = self.main_env.steps
        nb_steps = 0
        self.env.reset()
        self.env.set_state(
            self.main_env.agent_states, self.main_env.item_states, None, None, nb_steps
        )
        self.env.reset_history()
        print("reset env", self.env.get_state())
        # print(self.env.room)
        # for agent_id in range(self.num_agents):
        self.sim_env.reset()
        self.sim_env.set_state(
            self.env.agent_states, self.env.item_states, None, None, nb_steps
        )
        print("reset sim_env", self.sim_env.get_state())
        # pdb.set_trace()
        goals = [None] * self.num_agents  # TODO duplicates?
        Vs = [0] * self.num_agents
        Rs = [[] for _ in range(self.num_agents)]
        visitations = [[] for _ in range(self.num_agents)]
        selected_child_indexs = [
            [] for _ in range(self.num_agents)
        ]  # act idx in visitations. different than in action_space bc of randomiztion in MCTS!!!
        print(
            "plan_given_high_level_goals",
            self.main_env.get_full_state(),
            self.env.get_full_state(),
            self.main_env.running,
            self.env.running,
        )
        while self.env.running:
            print("self.goals", self.goals)
            currState = self.env.get_state()
            print("state:", [(state["pos"],state["angle"]) for state in currState])
            # pdb.set_trace()

            for agent_id, goal in enumerate(copy.deepcopy(self.goals)):
                # print('decomposition', self.planning_agent_id, goal)
                decomposed_goal = self.goal_decomposition(
                    self.sim_env, self.planning_agent_id, goal
                )

                self.env.goals[agent_id] = decomposed_goal

                for agent_id_tmp in range(self.num_agents):
                    self.sim_env.goals[agent_id] = decomposed_goal

                # print("subgoal:", agent_id, decomposed_goal)

            for agent_id, goal in enumerate(self.env.goals):
                if goals[agent_id] is None:
                    goals[agent_id] = [goal]
                else:
                    goals[agent_id].append(goal)

            self.sim_env.set_state(
                self.env.agent_states,
                self.env.item_states,
                None,
                None,
                self.main_env.steps,
            )

            mcts = MCTS(
                agent_id=self.planning_agent_id,
                action_space=self.sim_env.get_action_space(agent_id),
                transition=self._transition,  # return a method as func self._transition()self.sim_env.transition,
                # attached=self.sim_env.get_attr('attached')[0],
                reward=self.sim_env.get_reward_state,  # self._get_reward_state,
                is_terminal=self._terminal,  # self.sim_env.terminal,
                num_simulation=self.args.nb_simulations,
                max_rollout_steps=self.args.max_rollout_steps,
                c_init=self.args.cInit,
                c_base=self.args.cBase,
            )

            """TODO: check global environment status update & terminal condition"""
            terminal = False
            actions = [None for _ in range(self.num_agents)]

            for agent_id in range(self.num_agents):
                if self.goals[agent_id][0] is None:
                    actions[agent_id] = None
                    # print('self.goals',self.goals)
                    continue
                actions[agent_id], visitation, selected_child_index = select_action(
                    # agent_id,
                    # self.goals,
                    # self.env.get_attr('attached')[0],
                    self.env.steps,
                    self.sim_env,
                    mcts,
                    expected_action=None,  # for 2 agents
                )
                visitations[agent_id].append(visitation)
                selected_child_indexs[agent_id].append(selected_child_index)
                if actions[agent_id] is None:
                    terminal = True
                    break
                if terminal:
                    break
            if terminal:
                break

            print("plan_given_high_level_goals", actions)
            # pdb.set_trace()
            self.env.step(actions, planning_agent_id=self.planning_agent_id)
            print("self.env.running", self.env.running)
            print(self.env.get_state())
            # pdb.set_trace()

            rewards = [0] * self.num_agents

            for agent_id in range(self.num_agents):
                # print('self.goals rewards', self.goals)
                if self.goals[agent_id][0] is None:
                    continue
                rewards[agent_id] = self.env.get_reward_state(
                    agent_id,
                    self.env.get_state(),
                    "stop",
                    None,
                    None,
                    self.goals[agent_id].copy(),
                )
                Vs[agent_id] += rewards[agent_id]
                Rs[agent_id].append(rewards[agent_id])

            nb_steps += 1
            print(
                "self.env.steps",
                self.env.steps,
                "nb_steps",
                nb_steps,
                "self.max_episode_length",
                self.max_episode_length,
                "plan_length",
                plan_length,
            )
            # if self.env.steps >= self.max_episode_length or nb_steps >= plan_length:
            if self.env.steps >= plan_length:
                break

        plans = self.env.actions
        subgoals = goals
        trajs = self.env.trajectories
        Vs = Vs
        Rs = Rs
        return trajs, plans, selected_child_indexs, subgoals, Vs, Rs, visitations

    def plan(
        self,
        nb_episodes,
        sampled_subgoals=None,
        selected_subgoal_idx=None,
        record=False,
        save_demos=False,
        max_T=None,
    ):
        """sampled_subgoals: all goals that were sampled. self.goals are the selected ones to plan by.
        this function plans for 1 episode"""

        imitation_demos = []  # one demo for now.
        for episode_id in range(nb_episodes):
            self.episode_id = episode_id
            all_modes = self.rollout(record, episode_id, max_T=max_T)
            # save rollout in AIRL compatible format (for discriminator later) - 1 demo
            print("all_modes", all_modes)
            # print('episode_id',episode_id)
            agent_obs = all_modes[episode_id]["trajs"]
            obs = []
            for st in range(0, len(agent_obs[0]), self.env.NUM_STEPS_PER_TICK):
                curr_state = []
                for agent_id in range(self.env.num_agents):
                    curr_state += list(agent_obs[agent_id][st])  # tuple
                obs.append(np.array(curr_state))
            # pdb.set_trace()
            acts = all_modes[episode_id]["plans"][
                self.planning_agent_id
            ]  # string directions
            acts_idx = all_modes[episode_id]["plans_idx"][
                self.planning_agent_id
            ]  # act idx in visitation (not action_space)
            visitations = all_modes[episode_id]["visitation"][self.planning_agent_id]
            subgoals = all_modes[episode_id]["subgoals"][self.planning_agent_id]
            # pdb.set_trace()
            imitation_demos.append(
                types.TrajectoryWithRew(
                    obs=np.asarray(obs),  # format: (s_0,a_1,s_1,a_2,s_2,...,a_n,s_n)
                    acts=np.asarray(acts_idx),  # dummy values
                    infos=[
                        {
                            "goal": selected_subgoal_idx,  # should be state all self.envs are set in
                            "visitation": v,
                            "acts_idx": act_id,
                            "sampled_subgoals": sampled_subgoals,
                        }
                        for sg, v, act_id in zip(subgoals, visitations, acts_idx)
                    ],
                    rews=np.asarray(
                        all_modes[episode_id]["Rs"][self.planning_agent_id]
                    ),
                )
            )
        # print('imitation_demos', imitation_demos)
        if save_demos:
            types.save(
                os.path.join(self.args.log_dir, "expert_demos.pkl"), imitation_demos
            )
        return imitation_demos
