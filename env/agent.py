from gym import Env
from typing import Callable
from abc import ABC, abstractmethod
import torch
from tqdm import tqdm

from train.buffer import Buffer
from models.rssm import RSSM

class Policy(ABC):
    @abstractmethod
    def __call__(self, obs):
        pass

class RandomPolicy(Policy):
    def __init__(self, env: Env):
        self.env = env

    def __call__(self, obs):
        return self.env.action_space.sample()


class Agent:
    def __init__(self, env: Env, rssm: RSSM, buffer_size: int = 100000, collection_policy: str = "random", device="mps"):
        self.env = env
        match collection_policy:
            case "random":
                self.rollout_policy = RandomPolicy(env)
            case _:
                raise ValueError("Invalid rollout policy")

        self.buffer = Buffer(buffer_size, env.observation_space.shape, env.action_space.shape, device=device)
        self.rssm = rssm

    def data_collection_action(self, obs):
        return self.rollout_policy(obs)

    def collect_data(self, num_steps: int):
        obs = self.env.reset()
        done = False

        iterator = tqdm(range(num_steps), desc="Data Collection")
        for _ in iterator:
            action = self.data_collection_action(obs)
            next_obs, reward, done, _, _ = self.env.step(action)
            self.buffer.add(next_obs, action, reward, done)
            obs = next_obs
            if done:
                obs = self.env.reset()

