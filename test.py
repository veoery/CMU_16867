import numpy as np
import robosuite as suite
from lift2 import Lift2

# controller_config = load_part_controller_config(default_controller="JOINT_POSITION")

# from robosuite.models.objects import BoxObject

# box = BoxObject(name="box01", size=0.2).get_obj()

# create environment instance
# env = suite.make(
#     env_name="Lift",  # try with other tasks like "Stack" and "Door"
#     robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
#     has_renderer=True,
#     has_offscreen_renderer=False,
#     use_camera_obs=False,
# )

env = Lift2(
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)
print(env.observation_spec())
print(env.action_spec)


class Agent:
    def __init__(self, env: Lift2, learning_rate, initial_eps, final_eps, eps_dacay, discount_factor):
        self.env = env
        self.learning_rate = learning_rate
        self.initial_eps = initial_eps
        self.final_eps = final_eps
        self.discount_factor = discount_factor


# reset the environment
env.reset()

for i in range(1):
    action = np.random.randn(*env.action_spec[0].shape) * 0
    action[0] = 3.14
    # action[-1] = 0
    print()
    print(f"-----------------i={i}-----------------")
    print(action)
    # print(*env.action_spec)
    obs, reward, done, info = env.step(action)  # take action in the environment
    for k in obs:
        print(k, obs[k])
    # env.render()  # render on display
