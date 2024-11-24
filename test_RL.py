import numpy as np
import robosuite as suite
from lift2 import Lift2
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from robosuite.wrappers import GymWrapper

env = Lift2(
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    reward_shaping=True,  # Dense rewards for easier learning
)
# env = suite.make(
#     env_name="Lift",  # Task name
#     robots="Panda",  # Robot arm (e.g., Sawyer, Baxter also available)
#     has_renderer=True,  # Set to True for visual debugging
#     has_offscreen_renderer=False,
#     use_camera_obs=False,  # Use True if training with visual input
#     reward_shaping=True,  # Dense rewards for faster learning
# )

# Check observation and action specs
print("Observation Space:", env.observation_spec())
print("Action Space:", env.action_spec)

obs = env.reset()
print(f"initial obs = {obs}")

gym_env = GymWrapper(env)
assert isinstance(gym_env, GymWrapper), "Environment is not a GymWrapper"

# Create a vectorized environment
# vec_env = make_vec_env(lambda: env, n_envs=1)

# Initialize PPO
ppo_model = PPO(
    "MlpPolicy",  # Policy type (MLP for non-visual input)
    gym_env,  # Training environment
    verbose=1,  # Display training progress
    learning_rate=3e-4,  # Learning rate
    n_steps=2048,  # Number of steps per update
    batch_size=64,  # Training batch size
    gamma=0.99,  # Discount factor
    gae_lambda=0.95,  # Advantage estimation factor
)


# Train the agent
print(type(ppo_model))

print("begin training")
ppo_model.learn(total_timesteps=100000)

# Test the trained agent
print(type(ppo_model))
ppo_model.save("ppo_lift_model")

# Reset the environment
print("begin test")


def flatten_observation(obs_dict):
    relevant_keys = ["robot0_proprio-state", "object-state"]
    return np.concatenate([obs_dict[key].flatten() for key in relevant_keys])


env.close()

obs = env.reset()
obs_flat = flatten_observation(obs)
print(f"Initial observation shape: {obs_flat.shape}")

tim = 0
for _ in range(2000):
    action, info = ppo_model.predict(obs_flat)
    # action = np.random.randn(*env.action_spec[0].shape) * 0
    # print(f"Action taken: {action}")

    obs, reward, done, info = env.step(action)
    obs_flat = flatten_observation(obs)
    # print(f"New observation shape: {obs_flat.shape}")

    env.render()

    if done:
        tim += 1
        print(_)
        env.close()
        obs = env.reset()
        obs_flat = flatten_observation(obs)

print(tim)
