from stable_baselines3 import PPO
import numpy as np

class TrainedRL:
    def __init__(self, config=None):
        self.rl_agent = PPO.load("./src/best_model.zip")
        self.state = np.zeros((4,))  # CartPole expects 4D state

    def update_state(self, obs):
        self.state = np.array(obs)

    def update_action(self):
        action, _ = self.rl_agent.predict(self.state, deterministic=True)
        return int(action)
