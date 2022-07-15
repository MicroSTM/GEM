from typing import Optional

from gym.envs import register as gym_register
import pickle
import os

_ENTRY_POINT_PREFIX = "imitation.envs.examples.watchmove_envs"


def _register(env_name: str, entry_point: str, kwargs: Optional[dict] = None):
    entry_point = f"{_ENTRY_POINT_PREFIX}.{entry_point}"
    gym_register(id=env_name, entry_point=entry_point, kwargs=kwargs)


# trapezoid agents, circle targget objects
_register("WATCHMOVE-v0", entry_point="watchmove:WM")
_register("WATCHMOVETest-v0", entry_point="watchmove:WM_test")


task_dir = "../output/tasks/"
for task_file in os.listdir(task_dir):
    with open(os.path.join(task_dir, task_file), "rb") as input_file:
        data = pickle.load(input_file)
    if "nobj3" in task_file:
        _register(
            task_file[:-4] + "-v0",
            entry_point="watchmove:WM",
            kwargs={
                "G": data["G"],
                "predicates": data["predicates"],
                "expert_subgoals": data["expert_subgoals"],
                "init_positions": data["init_positions"],
                "shapes": data["shapes"],
                "colors": data["colors"],
                "num_agents": data["num_agents"],
            },
        )
    else:  # nobj4
        _register(
            task_file[:-4] + "-v0",
            entry_point="watchmove:WM",
            kwargs={
                "G": data["G"],
                "predicates": data["predicates"],
                "expert_subgoals": data["expert_subgoals"],
                "init_positions": data["init_positions"],
                "shapes": data["shapes"],
                "colors": data["colors"],
                "num_agents": data["num_agents"],
                "goals": [[None], [None], [None], [None]],
                "strengths": [0, 0, 0, 0],
                "sizes": [0, 0, 0, 0],
                "densities": [1, 1, 1, 1],
                "action_space_types": [0, 0, 0, 0],
                "costs": [0, 0, 0, 0],
                "temporal_decay": [0, 0, 0, 0],
                "visibility": [1, 1, 1, 1],
                "full_obs": [1, 1, 1, 1],
                "init_agent_angles": [0.0, 0.0, 0.0, 0.0],
            },
        )
