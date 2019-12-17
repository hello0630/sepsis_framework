import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """Basic autoencoder with one hidden layer before and after the encoding layer.

    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        """
        Args:
            input_dim (int): Dimension of the input (and reconstruction) layer.
            hidden_dim (int): Dimension of the hidden layer.
            latent_dim (int): Dimension of the latent space.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        encoding = torch.relu(self.fc2(h1))
        return encoding

    def decode(self, encoding):
        h3 = torch.relu(self.fc3(encoding))
        recon = torch.relu(self.fc4(h3))
        return recon

    def forward(self, x):
        encoding = self.encode(x)
        decoding = self.decode(encoding)
        return decoding
