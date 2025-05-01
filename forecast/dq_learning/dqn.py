import torch
import torch.nn.functional as F
from torch import nn


class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()

        # Define network layers
        self.fc1 = nn.Linear(in_states, h1_nodes)  # first fully connected layer
        self.out = nn.Linear(h1_nodes, out_actions)  # ouptut layer w

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply rectified linear unit (ReLU) activation
        x = self.out(x)  # Calculate output
        return x


if __name__ == "__main__":
    state_dim = 4
    action_dim = 3
    net = DQN(state_dim, 32, action_dim)
    state = torch.randn(10, state_dim)
    output = net(state)
    print(output)
