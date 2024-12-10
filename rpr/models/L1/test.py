import torch
from torch import nn
from ..data import X_test, y_test

print(torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


name = 'L1'
PATH = rf'src\models\{name}\{name}_full_best.pth'
model = torch.load(PATH, weights_only=False)
model.eval()


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, 0)
    return (preds == labels).type(torch.float).mean()
    

for x, y in zip(X_test, y_test):
    x, y = x.to(device), y.to(device)
    pred = model(x)
    print(f'Input: {x}')
    print(f'True output: {y}')
    print(f'Predict model: {pred}')
    print(f'MSE: {nn.MSELoss()(pred, y)}')
    print(f'Accuracy: {accuracy(pred, y)}')