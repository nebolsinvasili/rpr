import torch
from torch import nn, optim
from torchsummary import summary

from ..metrics import mean_absolute_error, mean_squared_error
from ..trainer import ModelTrainer
from .data import (
    X_test,
    X_train,
    X_valid,
    y_test,
    y_train,
    y_valid,
)
from .model import L3

print(torch.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


RANDOM_SEED = 42
EPOCHS = 500
BATCH_SIZE = 32
LR = 1e-2

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

name, model = "L3", L3().to(device)
summary(model, X_train.size())

MSE = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

trainer = ModelTrainer(
    model=model,
    loss_fn=MSE,
    optimizer=optimizer,
    metrics={
        "mean_absolute_error": mean_absolute_error,
        "mean_squared_error": mean_squared_error,
    },
    stoping=True,
    device=device,
    model_name=name,
    save_path=rf"rpr\models\{name}",
    lr_patience=3,
    lr_threshold=1e-3,
    stop_delta=0.05,
    stop_patience=10,
)
trainer.fit(
    train_data=(X_train, y_train),
    valid_data=(X_valid, y_valid),
    test_data=(X_test, y_test),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
)
