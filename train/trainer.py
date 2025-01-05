import numpy as np
import torch
import os
from torch.nn import functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from typing import Callable
import matplotlib.pyplot as plt
from gym import Env
from tqdm import tqdm
import logging

from models.rssm import *
from models.dynamics import DynamicsModel
from torch.utils.tensorboard import SummaryWriter
from env.agent import Agent
from env.envs import make_env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Output logs to console
        logging.FileHandler("training.log", mode="w")
    ]
)

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, rssm: RSSM, agent: Agent, optimizer: torch.optim.Optimizer, device: torch.device):
        self.rssm = rssm
        self.optimizer = optimizer
        self.device = device
        self.agent = agent

        self.writer = SummaryWriter()

    def collect_data(self, num_steps: int):
        self.agent.collect_data(num_steps)

    def save_buffer(self, path: str):
        self.agent.buffer.save(path)

    def train_batch(self, batch_size: int, seq_len: int, iteration: int, save_images: bool = False):
        obs, actions, rewards, dones = self.agent.buffer.sample(batch_size, seq_len)

        actions = torch.tensor(actions).long().to(self.device)
        actions = F.one_hot(actions, self.rssm.action_dim).float()

        obs = torch.tensor(obs, requires_grad=True).float().to(self.device)
        rewards = torch.tensor(rewards, requires_grad=True).float().to(self.device)
        dones = torch.tensor(dones).float().to(self.device)

        encoded_obs = self.rssm.encoder(obs.reshape(-1, *obs.shape[2:]).permute(0, 3, 1, 2))
        encoded_obs = encoded_obs.reshape(batch_size, seq_len, -1)

        rollout = self.rssm.generate_rollout(actions, obs=encoded_obs, dones=dones)

        hiddens, prior_states, posterior_states, prior_means, prior_logvars, posterior_means, posterior_logvars = rollout

        hiddens_reshaped = hiddens.reshape(batch_size * seq_len, -1)
        posterior_states_reshaped = posterior_states.reshape(batch_size * seq_len, -1)

        decoded_obs = self.rssm.decoder(hiddens_reshaped, posterior_states_reshaped)
        decoded_obs = decoded_obs.reshape(batch_size, seq_len, *obs.shape[-3:])

        reward_params = self.rssm.reward_model(hiddens, posterior_states)
        mean, logvar = torch.chunk(reward_params, 2, dim=-1)
        logvar = F.softplus(logvar)
        reward_dist = Normal(mean, torch.exp(logvar))
        predicted_rewards = reward_dist.rsample()

        if save_images:
            batch_idx = np.random.randint(0, batch_size)
            seq_idx = np.random.randint(0, seq_len - 3)
            fig = self._visualize(obs, decoded_obs, rewards, predicted_rewards, batch_idx,
                                  seq_idx, iteration, grayscale=True)
            if not os.path.exists("reconstructions"):
                os.makedirs("reconstructions")
            fig.savefig(f"reconstructions/iteration_{iteration}.png")
            self.writer.add_figure("Reconstructions", fig, iteration)
            plt.close(fig)

        reconstruction_loss = self._reconstruction_loss(decoded_obs, obs)
        kl_loss = self._kl_loss(prior_means, F.softplus(prior_logvars), posterior_means, F.softplus(posterior_logvars))
        reward_loss = self._reward_loss(rewards, predicted_rewards)

        loss = reconstruction_loss + kl_loss + reward_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.rssm.parameters(), 1, norm_type=2)
        self.optimizer.step()

        return loss.item(), reconstruction_loss.item(), kl_loss.item(), reward_loss.item()

    def train(self, iterations: int, batch_size: int, seq_len: int):
        self.rssm.train()
        iterator = tqdm(range(iterations), desc="Training", total=iterations)
        losses = []
        infos = []
        last_loss = float("inf")
        for i in iterator:
            loss, reconstruction_loss, kl_loss, reward_loss = self.train_batch(batch_size, seq_len, i,
                                                                               save_images=i % 100 == 0)

            self.writer.add_scalar("Loss", loss, i)
            self.writer.add_scalar("Reconstruction Loss", reconstruction_loss, i)
            self.writer.add_scalar("KL Loss", kl_loss, i)
            self.writer.add_scalar("Reward Loss", reward_loss, i)

            if loss < last_loss:
                self.rssm.save("rssm.pth")
                last_loss = loss

            info = {
                "Loss": loss,
                "Reconstruction Loss": reconstruction_loss,
                "KL Loss": kl_loss,
                "Reward Loss": reward_loss
            }
            losses.append(loss)
            infos.append(info)

            if i % 10 == 0:
                logger.info("\n----------------------------")
                logger.info(f"Iteration: {i}")
                logger.info(f"Loss: {loss:.4f}")
                logger.info(f"Running average last 20 losses: {sum(losses[-20:]) / 20: .4f}")
                logger.info(f"Reconstruction Loss: {reconstruction_loss:.4f}")
                logger.info(f"KL Loss: {kl_loss:.4f}")
                logger.info(f"Reward Loss: {reward_loss:.4f}")

    def _visualize(self, obs, decoded_obs, rewards, predicted_rewwards, batch_idx, seq_idx, iterations: int, grayscale: bool = True):
        obs = obs[batch_idx, seq_idx: seq_idx + 3]
        decoded_obs = decoded_obs[batch_idx, seq_idx: seq_idx + 3]

        rewards = rewards[batch_idx, seq_idx: seq_idx + 3]
        predicted_rewards = predicted_rewwards[batch_idx, seq_idx: seq_idx + 3]

        obs = obs.cpu().detach().numpy()
        decoded_obs = decoded_obs.cpu().detach().numpy()

        fig, axs = plt.subplots(3, 2)
        axs[0][0].imshow(obs[0, ..., 0], cmap="gray" if grayscale else None)
        axs[0][0].set_title(f"Iteration: {iterations} -- Reward: {rewards[0, 0]:.4f}")
        axs[0][0].axis("off")
        axs[0][1].imshow(decoded_obs[0, ..., 0], cmap="gray" if grayscale else None)
        axs[0][1].set_title(f"Pred. Reward: {predicted_rewards[0, 0]:.4f}")

        axs[0][1].axis("off")

        axs[1][0].imshow(obs[1, ..., 0], cmap="gray" if grayscale else None)
        axs[1][0].axis("off")
        axs[1][0].set_title(f"Reward: {rewards[1, 0]:.4f} ")
        axs[1][1].imshow(decoded_obs[1, ..., 0], cmap="gray" if grayscale else None)
        axs[1][1].set_title(f"Pred. Reward: {predicted_rewards[1, 0]:.4f}")
        axs[1][1].axis("off")

        axs[2][0].imshow(obs[2, ..., 0], cmap="gray" if grayscale else None)
        axs[2][0].axis("off")
        axs[2][0].set_title(f"Reward: {rewards[2, 0]:.4f}")
        axs[2][1].imshow(decoded_obs[2, ..., 0], cmap="gray" if grayscale else None)
        axs[2][1].set_title(f"Pred. Reward: {predicted_rewards[2, 0]:.4f}")
        axs[2][1].axis("off")

        return fig

    def _reconstruction_loss(self, decoded_obs, obs):
        return F.mse_loss(decoded_obs, obs)

    def _kl_loss(self, prior_means, prior_logvars, posterior_means, posterior_logvars):
        prior_dist = Normal(prior_means, torch.exp(prior_logvars))
        posterior_dist = Normal(posterior_means, torch.exp(posterior_logvars))

        return kl_divergence(posterior_dist, prior_dist).mean()

    def _reward_loss(self, rewards, predicted_rewards):
        return F.mse_loss(predicted_rewards, rewards)


if __name__ == "__main__":
    env = make_env("CarRacing-v2", render_mode="rgb_array", continuous=False, grayscale=True)
    hidden_size = 1024
    embedding_dim = 1024
    state_dim = 512

    encoder = EncoderCNN(in_channels=1, embedding_dim=embedding_dim)
    decoder = DecoderCNN(hidden_size=hidden_size, state_size=state_dim, embedding_size=embedding_dim,
                         output_shape=(1,128,128))
    reward_model = RewardModel(hidden_dim=hidden_size, state_dim=state_dim)
    dynamics_model = DynamicsModel(hidden_dim=hidden_size, state_dim=state_dim, action_dim=5, embedding_dim=embedding_dim)

    rssm = RSSM(dynamics_model=dynamics_model,
                encoder=encoder,
                decoder=decoder,
                reward_model=reward_model,
                hidden_dim=hidden_size,
                state_dim=state_dim,
                action_dim=5,
                embedding_dim=embedding_dim)

    optimizer = torch.optim.Adam(rssm.parameters(), lr=1e-3)
    agent = Agent(env, rssm)
    agent.buffer.load("buffer.npz")
    trainer = Trainer(rssm, agent, optimizer=optimizer, device="mps")
    trainer.collect_data(20000)
    trainer.save_buffer("buffer.npz")
    trainer.train(10000, 32, 20)