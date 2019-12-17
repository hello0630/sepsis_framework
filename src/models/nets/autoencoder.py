from definitions import *
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import MeanSquaredError, Loss
from sklearn.preprocessing import StandardScaler
from src.data.dicts import feature_dict
from src.models.nets.nets import Autoencoder


# CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    dataset = load_pickle(DATA_DIR + '/interim/preprocessed/dataset.dill')

    # Get vitals data
    X, _ = dataset.to_ml()
    X = X[:, dataset._col_indexer(['HR', 'O2Sat', 'SBP', 'MAP', 'DBP', 'Resp'])]

    # Perform scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = X_scaled[np.isnan(X_scaled).sum(axis=1) == 0]

    # Make loader
    torch_ds = TensorDataset(torch.Tensor(X_scaled), torch.Tensor(X_scaled))
    dataloader = DataLoader(torch_ds, batch_size=64, shuffle=True)

    # Setup
    model = Autoencoder(input_dim=X.shape[1], hidden_dim=30, latent_dim=2)
    optimizer = torch.optim.Adam(params=model.parameters())
    loss = nn.MSELoss()

    # Ignite me
    trainer = create_supervised_trainer(model, optimizer, loss)

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(trainer):
        print("Training results - EPOCH [{}]: Avg loss: {:.3f}".format(trainer.state.epoch, trainer.state.output))

    trainer.run(dataloader, max_epochs=2)
