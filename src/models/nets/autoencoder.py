from definitions import *
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import MeanSquaredError, Loss
from sklearn.preprocessing import StandardScaler
from src.data.dicts import feature_dict
from src.models.nets.nets import Autoencoder


# CUDA
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'


if __name__ == '__main__':
    dataset = load_pickle(DATA_DIR + '/interim/preprocessed/dataset.dill')

    # Get vitals data
    X, _ = dataset.to_ml()
    X = X[:, dataset._col_indexer(['HR', 'O2Sat', 'SBP', 'MAP', 'DBP', 'Resp'])]
    # X = X[:, dataset._col_indexer(['Glucose', 'SaO2'])]
    del dataset

    # Perform scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = X_scaled[np.isnan(X_scaled).sum(axis=1) == 0]
    print(X_scaled.shape)
    del X

    # Make loader
    torch_ds = TensorDataset(torch.Tensor(X_scaled), torch.Tensor(X_scaled))
    num_split = int(X_scaled.shape[0] / 4)
    train_dataset, val_dataset = random_split(torch_ds, [X_scaled.shape[0] - num_split, num_split])
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)

    # Setup
    model = Autoencoder(input_dim=X_scaled.shape[1], hidden_dim=100, latent_dim=3).to(device)
    optimizer = torch.optim.Adam(params=model.parameters())
    loss = nn.MSELoss()

    # Ignite me
    trainer = create_supervised_trainer(model, optimizer, loss, device=device)
    evaluator = create_supervised_evaluator(model, metrics={'mse': MeanSquaredError()}, device=device)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_loss(trainer):
        print("Training results - EPOCH [{}]: Avg loss: {:.3f}".format(trainer.state.epoch, trainer.state.output))

    @trainer.on(Events.EPOCH_STARTED)
    def validation_loss(trainer):
        if trainer.state.epoch == 1:
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            print("Validation Results - Epoch: {}  Avg MSE: {:.2f}"
                  .format(trainer.state.epoch, metrics['mse']))

    @trainer.on(Events.EPOCH_COMPLETED)
    def validation_loss(trainer):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print("Validation Results - Epoch: {}  Avg MSE: {:.2f}"
              .format(trainer.state.epoch, metrics['mse']))

    trainer.run(train_loader, max_epochs=5)

    torch.save(model.state_dict(), MODELS_DIR + '/nets/autoencoders/vitals.pt')

