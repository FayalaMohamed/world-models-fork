import torch
import torch.nn as nn
import torch.functional as F
from typing import Tuple

class EncoderCNN(nn.Module):
    def __init__(self, in_channels: int, embedding_dim: int = 2048, input_shape: Tuple[int, int] = (128, 128)):
        super(EncoderCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.fc1 = nn.Linear(self._compute_conv_output((in_channels, input_shape[0], input_shape[1])), embedding_dim)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

    def _compute_conv_output(self, shape: Tuple[int, int, int]):
        with torch.no_grad():
            x = torch.randn(1, shape[0], shape[1], shape[2])
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)

            return x.shape[1] * x.shape[2] * x.shape[3]


    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.bn1(x)
        x = torch.relu(self.conv2(x))
        x = self.bn2(x)

        x = torch.relu(self.conv3(x))
        x = self.bn3(x)

        x = self.conv4(x)
        x = self.bn4(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x


class DecoderCNN(nn.Module):
    def __init__(self, hidden_size: int, state_size: int,  embedding_size: int,
                 use_bn: bool = True, output_shape: Tuple[int, int] = (3, 128, 128)):
        super(DecoderCNN, self).__init__()

        self.output_shape = output_shape

        self.embedding_size = embedding_size
        self.fc1 = nn.Linear(hidden_size + state_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, 256 * (output_shape[1] // 16) * (output_shape[2] // 16))

        self.conv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)  # ×2
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)  # ×2
        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)  # ×2
        self.conv4 = nn.ConvTranspose2d(32, output_shape[0], kernel_size=3, stride=2, padding=1, output_padding=1)

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)

        self.use_bn = use_bn


    def forward(self, h: torch.Tensor, s: torch.Tensor):
        x = torch.cat([h, s], dim=-1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)

        x = x.view(-1, 256, self.output_shape[1] // 16, self.output_shape[2] // 16)

        if self.use_bn:
            x = torch.relu(self.bn1(self.conv1(x)))
            x = torch.relu(self.bn2(self.conv2(x)))
            x = torch.relu(self.bn3(self.conv3(x)))

        else:
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))

        x = self.conv4(x)

        return x



        return h_next, c_next

    def init_hidden(self, batch_size, height, width):
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
        )


class RewardModel(nn.Module):
    def __init__(self, hidden_dim: int, state_dim: int):
        super(RewardModel, self).__init__()

        self.fc1 = nn.Linear(hidden_dim + state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2)

    def forward(self, h: torch.Tensor, s: torch.Tensor):
        x = torch.cat([h, s], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x





if __name__ == "__main__":
    encoder = EncoderCNN(3)
    decoder = DecoderCNN(1024, 30, 16384)


    x = torch.randn(2, 3, 128, 128)
    s = torch.randn(2, 30)
    h = torch.randn(2, 1024)
    out = decoder(s, h)
    print(out.shape)
