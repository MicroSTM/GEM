# generate expert demos with a script

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import argparse
import timeit
import gym
import pdb
from datetime import datetime

from imitation.planner.planner_v1.MCTS_agent import *

import imitation.util.sacred as sacred_util
from imitation.data import rollout, types
from imitation.policies import serialize
from imitation.rewards.serialize import load_reward
from imitation.scripts.config.expert_demos import expert_demos_ex
from imitation.util import logger, util  # need only util, load in this order o/w error

parser = argparse.ArgumentParser()
parser.add_argument("--env-name", type=str)  # --env-name PHASE-v0
parser.add_argument("--seed", type=int, default=-1)
parser.add_argument("--log-dir", type=str)
parser.add_argument("--num-vec", type=int, default=1)
parser.add_argument(
    "--nb-simulations", type=int, default=1000, help="Number of MCTS simulations"
)  # 1000
parser.add_argument(
    "--max-rollout-steps",
    type=int,
    default=10,
    help="Maximum number of rollout steps in a simulation",
)
parser.add_argument("--cInit", type=float, default=1.25, help="Hyper-param for MCTS")
parser.add_argument("--cBase", type=float, default=1000, help="Hyper-param for MCTS")
# parser.add_argument(
#     "--max-episode-length",
#     type=int,
#     default=60,
#     metavar="LENGTH",
#     help="Maximum episode length",
# )  # 60
parser.add_argument(
    "--execute-length", type=int, default=20, help="Length of plans that get executed"
)  # 20
parser.add_argument(
    "--planning-agent-id", type=int, default=0
)  # objects 0,1,2. put object 0 on pos near object 1.
parser.add_argument("--radius", type=int, default=8)
parser.add_argument("--nsamples-per-ag", type=int, default=None)
parser.add_argument("--max-episode-length", type=int, default=20)
parser.add_argument("--plan-length", type=int, default=20)

if __name__ == "__main__":

    args = parser.parse_args()

    if args.seed == -1:
        random.seed()
        np.random.seed()
    else:
        random.seed(args.seed)
        np.random.seed(args.seed)

    args.main_env = gym.make(args.env_name)
    args.main_env.reset()
    args.env = gym.make(args.env_name)
    args.sim_env = gym.make(args.env_name)

    active_demo_obj_ids = [exp_goal[1] for exp_goal in args.main_env.expert_subgoals]
    args.nsamples_per_ag = len(args.main_env._action_space) * len(active_demo_obj_ids)
    
    full_imitation_demo = None

    for subgoal in args.main_env.expert_subgoals:
        print('planning for subgoal ', subgoal)
        args.planning_agent_id = subgoal[1]
        goals = [[None] for _ in range(args.main_env.num_agents)]    
        goals[args.planning_agent_id] = [subgoal[0], args.planning_agent_id, subgoal[2], 1] 
        args.main_env.goals = copy.deepcopy(goals)
        args.env.goals = copy.deepcopy(goals)
        args.sim_env.goals = copy.deepcopy(goals) 
        args.goals = copy.deepcopy(goals)

        """important that same scheme as in planner/select_subgoal_and_plan for AIRL discrim loss"""        
        min_x, min_y, max_x, max_y = args.main_env.get_boundaries()
        st_states = args.main_env.agent_states
        st_obs = args.main_env.get_full_state()
        n_entities = args.main_env.num_agents
        len_state = len(st_obs) // n_entities
        sampled_pos, sampled_object_id, sampled_subgoal_states = [], [], []
        # for each entity sample pos in some radius [sample 100 of these subgoals]
        # for agent_id in range(n_entities):
        # for agent_id in range(1): #sample 'fake subgoals' for only agent 0

        for agent_id in [args.planning_agent_id]:
            nsamples_per_arg_cntr = 0
            while nsamples_per_arg_cntr < args.nsamples_per_ag:
                x = st_states[agent_id]["pos"][0] + random.choice([-1, 1]) * (
                    round(np.random.uniform(1, args.radius), 2)
                )
                y = st_states[agent_id]["pos"][1] + random.choice([-1, 1]) * (
                    round(np.random.uniform(1, args.radius), 2)
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
            # only 1 goal should be not None when done is composed of completing multiple subgoals..
            if args.goals[agent_id][0] is not None:
                goal_state = st_obs.copy()
                if args.goals[agent_id][0] == 'LMA':
                    x, y = args.goals[agent_id][2]
                    goal_state[agent_id * len_state] = x
                    goal_state[(agent_id * len_state) + 1] = y
                elif args.goals[agent_id][0] == 'ROT':
                    a = args.goals[agent_id][2]
                    goal_state[(agent_id * len_state) + 2] = a
                sampled_subgoal_states[-1] = goal_state
                selected_idx = len(sampled_subgoal_states) - 1
        # pdb.set_trace() #check sampled_subgoal_states, selected_idx

        agent = MCTS_agent(args)
        # more than 1 predicate --> more than 1 plan call
        imitation_demo = agent.plan(
            nb_episodes=1,
            sampled_subgoals=sampled_subgoal_states,
            selected_subgoal_idx=selected_idx,
            save_demos=False,
        )
        args.main_env.running = True
        args.env.running = True

        if full_imitation_demo is None:
            full_imitation_demo = imitation_demo
        else:            
            obs = np.concatenate([full_imitation_demo[0].obs, imitation_demo[0].obs[1:]]) # format: (s_0,a_1,s_1,a_2,s_2,...,a_n,s_n)
            acts = np.concatenate([full_imitation_demo[0].acts, imitation_demo[0].acts])
            rews = np.concatenate([full_imitation_demo[0].rews, imitation_demo[0].rews])
            infos = full_imitation_demo[0].infos + imitation_demo[0].infos
            # pdb.set_trace()
            full_imitation_demo = [types.TrajectoryWithRew(
                                    obs=obs,
                                    acts=acts,
                                    infos=infos,
                                    rews=rews,
                                )]    
                
    types.save(os.path.join(args.log_dir,'expert_demos.pkl'), full_imitation_demo)
