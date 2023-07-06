from torch import nn
import torch


class SimpleSDFModel(nn.Module):
    def __init__(self, n_input, n_output, n_hidden) -> None:
        super().__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.fc = nn.Sequential(
            nn.Linear(self.n_input, self.n_hidden),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.Linear(self.n_hidden, self.n_output),
        )

    def forward(self, x):
        x = torch.sigmoid(self.fc(x))
        return x
