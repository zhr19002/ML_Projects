import os
import re
import gymnasium as gym
import ale_py
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3 import PPO

ENV_ID = "ALE/DemonAttack-v5"
LOG_DIR = "./logs/"
FINAL_MODEL_PATH = os.path.join(LOG_DIR, "final_PPO_model.zip")
CHECKPOINT_PREFIX = "PPO_model"
TOTAL_TIMESTEPS = 10_000_000

os.makedirs(LOG_DIR, exist_ok=True)

gym.register_envs(ale_py)

def make_env():
    env = gym.make(ENV_ID, render_mode="rgb_array")
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    return Monitor(env)

env = DummyVecEnv([make_env])
env = VecTransposeImage(env)
env = VecFrameStack(env, n_stack=4)

eval_env = DummyVecEnv([make_env])
eval_env = VecTransposeImage(eval_env)
eval_env = VecFrameStack(eval_env, n_stack=4)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=LOG_DIR,
    log_path=LOG_DIR,
    eval_freq=10000,
    n_eval_episodes=10,
    deterministic=True,
    render=False
)

checkpoint_callback = CheckpointCallback(
    save_freq=100000,
    save_path=LOG_DIR,
    name_prefix=CHECKPOINT_PREFIX
)

def find_latest_checkpoint(log_dir, prefix):
    checkpoint_files = [
        f for f in os.listdir(log_dir) if f.startswith(prefix) and f.endswith(".zip")
    ]
    if not checkpoint_files:
        return None

    checkpoint_files.sort(
        key=lambda x: int(re.findall(rf"{prefix}_(\d+)_steps", x)[0]),
        reverse=True
    )
    return os.path.join(log_dir, checkpoint_files[0])

checkpoint_path = find_latest_checkpoint(LOG_DIR, CHECKPOINT_PREFIX)
model = None

if checkpoint_path:
    print(f"Restore checkpoint: {checkpoint_path}")
    model = PPO.load(checkpoint_path, env=env)
elif os.path.exists(FINAL_MODEL_PATH):
    print("Restore final model: final_PPO_model.zip")
    model = PPO.load(FINAL_MODEL_PATH, env=env)
else:
    print("No model detected, start training: ")
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        clip_range=0.2,
        tensorboard_log=LOG_DIR
    )

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[eval_callback, checkpoint_callback],
    reset_num_timesteps=False
)

model.save(FINAL_MODEL_PATH)