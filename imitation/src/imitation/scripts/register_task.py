from typing import Optional
from gym.envs import register as gym_register
import argparse
import pickle
import pdb

_ENTRY_POINT_PREFIX = "imitation.envs.examples.watchmove_envs"

parser = argparse.ArgumentParser(description="register task")
parser.add_argument("--task-path", default="")
parser.add_argument("--task-name", default="")
args = parser.parse_args()


def _register(env_name: str, entry_point: str, kwargs: Optional[dict] = None):
    entry_point = f"{_ENTRY_POINT_PREFIX}.{entry_point}"
    gym_register(id=env_name, entry_point=entry_point, kwargs=kwargs)


with open(args.task_path, "rb") as input_file:
    data = pickle.load(input_file)

# pdb.set_trace()
_register(
    args.task_name,
    entry_point="WATCHMOVE:WM",
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
