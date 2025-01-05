import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicsModel(nn.Module):
    def __init__(self, hidden_dim: int, action_dim: int, state_dim: int, embedding_dim: int, rnn_layer: int = 1):
        super(DynamicsModel, self).__init__()

        self.hidden_dim = hidden_dim

        self.rnn = nn.ModuleList([nn.GRUCell(hidden_dim, hidden_dim) for _ in range(rnn_layer)])
        self.project_state_action = nn.Linear(action_dim + state_dim, hidden_dim)

        self.prior = nn.Linear(hidden_dim, state_dim * 2)
        self.project_hidden_action = nn.Linear(hidden_dim + action_dim, hidden_dim)

        self.posterior = nn.Linear(hidden_dim, state_dim * 2)
        self.project_hidden_obs = nn.Linear(hidden_dim + embedding_dim, hidden_dim)

        self.state_dim = state_dim

        self.act_fn = nn.ReLU()

    def forward(self, prev_hidden: torch.Tensor, prev_state: torch.Tensor, actions: torch.Tensor,
                obs: torch.Tensor = None, dones: torch.Tensor = None):
        """
        Forward pass of the dynamics model for one time step.
        :param prev_hidden: Previous hidden state of the RNN: (batch_size, hidden_dim)
        :param prev_state: Previous stochastic state: (batch_size, state_dim)
        :param action: One hot encoded actions: (sequence_length, batch_size, action_dim)
        :param obs: This is the encoded observation from the encoder, not the raw observation!: (sequence_length, batch_size, embedding_dim)
        :return:
        """
        B, T, _ = actions.size()

        hiddens_list = []
        posterior_means_list = []
        posterior_logvars_list = []
        prior_means_list = []
        prior_logvars_list = []
        prior_states_list = []
        posterior_states_list = []

        hiddens_list.append(prev_hidden.unsqueeze(1))  # (B, 1, hidden_dim)
        prior_states_list.append(prev_state.unsqueeze(1))
        posterior_states_list.append(prev_state.unsqueeze(1))

        for t in range(T - 1):
            ### Combine the state and action ###
            action_t = actions[:, t, :]
            obs_t = obs[:, t, :] if obs is not None else torch.zeros(B, self.embedding_dim, device=actions.device)
            state_t = posterior_states_list[-1][:, 0, :] if obs is not None else prior_states_list[-1][:, 0, :]
            state_t = state_t if dones is None else state_t * (1 - dones[:, t, :])
            hidden_t = hiddens_list[-1][:, 0, :]

            state_action = torch.cat([state_t, action_t], dim=-1)
            state_action = self.act_fn(self.project_state_action(state_action))

            ### Update the deterministic hidden state ###
            for i in range(len(self.rnn)):
                hidden_t = self.rnn[i](state_action, hidden_t)

            ### Determine the prior distribution ###
            hidden_action = torch.cat([hidden_t, action_t], dim=-1)
            hidden_action = self.act_fn(self.project_hidden_action(hidden_action))
            prior_params = self.prior(hidden_action)
            prior_mean, prior_logvar = torch.chunk(prior_params, 2, dim=-1)

            ### Sample from the prior distribution ###
            prior_dist = torch.distributions.Normal(prior_mean, torch.exp(F.softplus(prior_logvar)))
            prior_state_t = prior_dist.rsample()

            ### Determine the posterior distribution ###
            if obs is None:
                posterior_mean = prior_mean
                posterior_logvar = prior_logvar
            else:
                hidden_obs = torch.cat([hidden_t, obs_t], dim=-1)
                hidden_obs = self.act_fn(self.project_hidden_obs(hidden_obs))
                posterior_params = self.posterior(hidden_obs)
                posterior_mean, posterior_logvar = torch.chunk(posterior_params, 2, dim=-1)

            ### Sample from the posterior distribution ###
            posterior_dist = torch.distributions.Normal(posterior_mean, torch.exp(F.softplus(posterior_logvar)))
            posterior_state_t = posterior_dist.rsample()

            ### Store results in lists (instead of in-place modification) ###
            posterior_means_list.append(posterior_mean.unsqueeze(1))
            posterior_logvars_list.append(posterior_logvar.unsqueeze(1))
            prior_means_list.append(prior_mean.unsqueeze(1))
            prior_logvars_list.append(prior_logvar.unsqueeze(1))
            prior_states_list.append(prior_state_t.unsqueeze(1))
            posterior_states_list.append(posterior_state_t.unsqueeze(1))
            hiddens_list.append(hidden_t.unsqueeze(1))

        # Convert lists to tensors using torch.cat()
        hiddens = torch.cat(hiddens_list, dim=1)
        prior_states = torch.cat(prior_states_list, dim=1)
        posterior_states = torch.cat(posterior_states_list, dim=1)
        prior_means = torch.cat(prior_means_list, dim=1)
        prior_logvars = torch.cat(prior_logvars_list, dim=1)
        posterior_means = torch.cat(posterior_means_list, dim=1)
        posterior_logvars = torch.cat(posterior_logvars_list, dim=1)

        return hiddens, prior_states, posterior_states, prior_means, prior_logvars, posterior_means, posterior_logvars


if __name__ == "__main__":
    dynamics = DynamicsModel(hidden_dim=128, action_dim=3, state_dim=256, embedding_dim=128)
    hidden = torch.randn(2, 128)
    state = torch.randn(2, 256)
    actions = torch.randn(2, 10, 3)
    obs = torch.randn(2, 10, 128)

    hiddens, prior_states, posterior_states, prior_means, prior_logvars, posterior_means, posterior_logvars = dynamics(
        hidden, state, actions, obs)
