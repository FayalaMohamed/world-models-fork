import torch
import numpy as np


class Buffer:
    def __init__(self, buffer_size: int, obs_shape: tuple, action_shape: tuple, device: torch.device):
        self.buffer_size = buffer_size
        self.obs_buffer = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self.action_buffer = np.zeros((buffer_size, *action_shape), dtype=np.int32)  # Ensure integer type
        self.reward_buffer = np.zeros((buffer_size, 1), dtype=np.float32)
        self.done_buffer = np.zeros((buffer_size, 1), dtype=np.bool_)

        self.device = device

        self.idx = 0

    def add(self, obs: torch.Tensor, action: int, reward: float, done: bool):
        self.obs_buffer[self.idx] = obs
        self.action_buffer[self.idx] = action
        self.reward_buffer[self.idx] = reward
        self.done_buffer[self.idx] = done

        self.idx = (self.idx + 1) % self.buffer_size


    def sample(self, batch_size: int, sequence_length: int):
        starting_idxs = np.random.randint(0, (self.idx % self.buffer_size) - sequence_length, (batch_size,))

        index_tensor = np.stack([np.arange(start, start + sequence_length) for start in starting_idxs])
        obs_sequence = self.obs_buffer[index_tensor]
        action_sequence = self.action_buffer[index_tensor]
        reward_sequence = self.reward_buffer[index_tensor]
        done_sequence = self.done_buffer[index_tensor]

        return obs_sequence, action_sequence, reward_sequence, done_sequence


    def save(self, path: str):
        np.savez(path, obs_buffer=self.obs_buffer, action_buffer=self.action_buffer,
                 reward_buffer=self.reward_buffer, done_buffer=self.done_buffer, idx=self.idx)

    def load(self, path: str):
        data = np.load(path)
        self.obs_buffer = data["obs_buffer"]
        self.action_buffer = data["action_buffer"]
        self.reward_buffer = data["reward_buffer"]
        self.done_buffer = data["done_buffer"]
        self.idx = data["idx"]
