BASE_DIR="../output"
TASK_VERSION="-v0"
GPU_DEVICE=0
NOBJ=3
TASK_ID=2
TASK_BASE='nobj'$NOBJ'_taskid'
TASK_NAME='train_'$TASK_BASE$TASK_ID$TASK_VERSION
TEST_BASE='test_nobj'$NOBJ'_taskid'
TEST_ID=0
TEST_NAME=$TEST_BASE$TASK_ID'_'$TEST_ID$TASK_VERSION     
seed=12

#M-AIRL note: set expert_batch_size to the length of the expert demo
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -m imitation.scripts.train_adversarial with algorithm=airl env_name=$TASK_NAME \
rollout_path=$BASE_DIR/expert_demos/$TASK_NAME/expert_demos.pkl \
log_dir=$BASE_DIR/GEM/$TASK_NAME/$seed \
total_timesteps=500000 use_action=False use_graph=True \
use_attention=False fc_G=True reward_type='GNN' expert_batch_size=12 policy_type='planner' \
beta=0.3

#Active reward learning
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -m imitation.scripts.gen_query_trans_set_state with env_name=$TASK_NAME render=False videos=False \
log_dir=$BASE_DIR/GEM/$TASK_NAME/$seed/gen_query_trans_set_state/ \
policy_path=$BASE_DIR/GEM/$TASK_NAME/$seed/checkpoints/final/gen_policy/ \
reward_type="RewardNet_unshaped" \
reward_dir=$BASE_DIR/GEM/$TASK_NAME/$seed/checkpoints/final/ \
rollout_path=$BASE_DIR/expert_demos//$TASK_NAME/expert_demos.pkl \
global_trans=True prev_reward=True add_new_state=True alpha=0.0 beta=0.3 only_add_transforms=False min_num_transform_iters=0 \
prob_trans=0.2 prop_prob_reduce=0.5 query_episode_steps=1 num_iters=300 num_opt_steps=5000 \
seed_path=$BASE_DIR/GEM/$TASK_NAME/$seed/sacred/config.json

#Testing
CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -m imitation.scripts.eval_set_state with env_name=$TEST_NAME render=False videos=False \
log_dir=$BASE_DIR/GEM/$TASK_NAME/$seed/eval_policy/$TEST_ID \
reward_type="RewardNet_unshaped" \
reward_dir=$BASE_DIR/GEM/$TASK_NAME/$seed/checkpoints/final/ \
rollout_path=$BASE_DIR/expert_demos/$TASK_NAME/expert_demos.pkl \
model_dir=$BASE_DIR/GEM/$TASK_NAME/$seed/gen_query_trans_set_state \
global_trans=True prev_reward=True add_new_state=True alpha=0.0 beta=0.3 only_add_transforms=False min_num_transform_iters=0 \
prob_trans=0.2 prop_prob_reduce=0.5 query_episode_steps=1 num_iters=300 \
seed_path=$BASE_DIR/GEM/$TASK_NAME/$seed/sacred/config.json


