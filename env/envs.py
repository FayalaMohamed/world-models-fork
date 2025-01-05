import torch
import numpy as np
import gym
import cv2 as cv
from typing import Union, Sequence, Tuple
import collections

from gym.core import ActType, ObsType


class InitialWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, no_ops: int = 0, repeat: int = 1):
        super(InitialWrapper, self).__init__(env)
        self.repeat = repeat
        self.no_ops = no_ops
        self.repeat = repeat

        self.op_counter = 0

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        if self.op_counter < self.no_ops:
            obs, reward, done, info = self.env.step(0)
            self.op_counter += 1

        total_reward = 0.0
        done = False
        for _ in range(self.repeat):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break

        return obs, total_reward, done, info


class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, new_shape: Sequence[int] = (128, 128, 3), grayscale: bool = False):
        super(PreprocessFrame, self).__init__(env)
        self.shape = new_shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=self.shape, dtype=np.float32)
        self.grayscale = grayscale

        if self.grayscale:
            self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(*self.shape[:-1], 1), dtype=np.float32)

    def observation(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.astype(np.uint8)
        new_frame = cv.resize(obs, self.shape[:-1], interpolation=cv.INTER_AREA)
        if self.grayscale:
            new_frame = cv.cvtColor(new_frame, cv.COLOR_RGB2GRAY)
            new_frame = np.expand_dims(new_frame, -1)

        torch_frame = torch.from_numpy(new_frame).float()
        torch_frame = torch_frame / 255.0

        return torch_frame

def make_env(env_name: str, new_shape: Sequence[int] = (128, 128, 3), grayscale: bool = True, **kwargs):
    env = gym.make(env_name, **kwargs)
    env = PreprocessFrame(env, new_shape, grayscale=grayscale)
    return env

if __name__ == "__main__":
    env = make_env("CarRacing-v2", render_mode="rgb_array", continuous=False)
    print(env.observation_space.shape)
