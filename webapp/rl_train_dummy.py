import gymnasium as gym
import sys
sys.modules['gym'] = gym
gym.__version__ = "0.999"  # prevent SB3 save error

from stable_baselines3 import PPO

# Create a simple environment
env = gym.make("CartPole-v1")

# Create and train the agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Save the trained model
model.save("./src/best_model")  # will create best_model.zip
print("✅ Model trained and saved as ./src/best_model.zip")
