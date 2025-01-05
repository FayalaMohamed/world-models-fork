import torch
import torch.nn as nn
import torch.distributions as dist

from models.models import EncoderCNN, DecoderCNN, RewardModel


class RSSM:
    def __init__(self,
                 encoder: EncoderCNN,
                 decoder: DecoderCNN,
                 reward_model: RewardModel,
                 dynamics_model: nn.Module,
                 hidden_dim: int,
                 state_dim: int,
                 action_dim: int,
                 embedding_dim: int,
                 device: str = "mps"):
        """
        Recurrent State-Space Model (RSSM) for learning dynamics models.

        Args:
            encoder: Encoder network for deterministic state
            prior_model: Prior network for stochastic state
            decoder: Decoder network for reconstructing observation
            sequence_model: Recurrent model for deterministic state
            hidden_dim: Hidden dimension of the RNN
            latent_dim: Latent dimension of the stochastic state
            action_dim: Dimension of the action space
            obs_dim: Dimension of the encoded observation space


        """
        super(RSSM, self).__init__()

        self.dynamics = dynamics_model
        self.encoder = encoder
        self.decoder = decoder
        self.reward_model = reward_model

        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim

        #shift to device
        self.dynamics.to(device)
        self.encoder.to(device)
        self.decoder.to(device)
        self.reward_model.to(device)


    def generate_rollout(self, actions: torch.Tensor, hiddens: torch.Tensor = None, states: torch.Tensor = None,
                         obs: torch.Tensor = None, dones: torch.Tensor = None):

        if hiddens is None:
            hiddens = torch.zeros(actions.size(0), self.hidden_dim).to(actions.device)

        if states is None:
            states = torch.zeros(actions.size(0), self.state_dim).to(actions.device)

        dynamics_result = self.dynamics(hiddens, states, actions, obs, dones)
        hiddens, prior_states, posterior_states, prior_means, prior_logvars, posterior_means, posterior_logvars = dynamics_result

        return hiddens, prior_states, posterior_states, prior_means, prior_logvars, posterior_means, posterior_logvars

    def train(self):
        self.dynamics.train()
        self.encoder.train()
        self.decoder.train()
        self.reward_model.train()

    def eval(self):
        self.dynamics.eval()
        self.encoder.eval()
        self.decoder.eval()
        self.reward_model.eval()

    def encode(self, obs: torch.Tensor):
        return self.encoder(obs)

    def decode(self, state: torch.Tensor):
        return self.decoder(state)

    def predict_reward(self, h: torch.Tensor, s: torch.Tensor):
        return self.reward_model(h, s)

    def parameters(self):
        return list(self.dynamics.parameters()) + list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.reward_model.parameters())

    def save(self, path: str):
        torch.save({
            "dynamics": self.dynamics.state_dict(),
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "reward_model": self.reward_model.state_dict()
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path)
        self.dynamics.load_state_dict(checkpoint["dynamics"])
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.decoder.load_state_dict(checkpoint["decoder"])
        self.reward_model.load_state_dict(checkpoint["reward_model"])

if __name__ == "__main__":
    pass
