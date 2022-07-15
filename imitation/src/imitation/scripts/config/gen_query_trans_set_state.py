import os

import sacred

from imitation.util import util

gen_query_trans_set_state_ex = sacred.Experiment("gen_query_trans_set_state")


@gen_query_trans_set_state_ex.config
def replay_defaults():
    env_name = "CartPole-v1"  # environment to evaluate in
    eval_n_timesteps = None  # Min timesteps to evaluate, optional.
    eval_n_episodes = 1  # Num episodes to evaluate, optional.
    query_n_samples = 1  # Number of samples for generating query
    init_episode_steps = 1  # Number of steps in the initial rollout
    query_episode_steps = 1  # Number of steps in a query
    prev_reward = False  # Using previous reward
    num_vec = 1  # Number of environments in parallel
    nsamples_per_ag = 1000  # number of sampled subgoals
    parallel = False  # Use SubprocVecEnv (generally faster if num_vec>1)
    max_episode_steps = None  # Set to positive int to limit episode horizons
    expert_batch_size = 16

    normalize_state = False
    global_trans = False
    min_num_transform_iters = 8
    num_transforms = 2
    only_add_transforms = False
    prob_trans = 0.5
    prop_prob_reduce = 0.8  # Probability of removing an edge
    max_num_attempts = 100000  # Number of attempts to sample a new graph
    num_opt_steps = 5000  # Number of optimization steps
    alpha = 0.1  # Coefficient for the diversity term
    w_prev = 0.2  # Coefficient for the previous reward term
    beta = 1.0
    num_iters = 100  # Number of iterations
    early_stopping_thresh = (
        0.01  # Early stopping criterion for finetuning the reward func
    )
    add_new_states = False
    exclude = False

    videos = False  # save video files
    video_kwargs = {}  # arguments to VideoWrapper
    render = True  # render to screen
    render_fps = 60  # -1 to render at full speed
    log_root = os.path.join("output", "gen_query")  # output directory

    policy_type = "ppo"  # class to load policy, see imitation.policies.loader
    policy_path = (  # serialized policy
        "tests/data/expert_models/cartpole_0/policies/final/"
    )

    reward_type = None  # Optional: override with reward of this type
    reward_path = None  # Path of serialized reward to load


@gen_query_trans_set_state_ex.config
def logging(log_root, env_name):
    log_dir = os.path.join(
        log_root, env_name.replace("/", "_"), util.make_unique_timestamp()
    )
    rollout_save_path = os.path.join(log_dir, "rollout.pkl")


@gen_query_trans_set_state_ex.named_config
def fast():
    eval_n_timesteps = 1
    eval_n_episodes = None
    max_episode_steps = 1
