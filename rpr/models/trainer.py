import os
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm


class EarlyStopping:
    def __init__(self, min_delta=50, patience=10):
        """
        Args:
            min_delta (float): Минимальное изменение ошибки, приемлемое как улучшение.
            patience (int): Количество эпох без улучшений после которых обучение будет остановлено.
        """
        self.min_delta = min_delta
        self.patience = patience
        self.patience_counter = 0
        self.best_loss = None

    def __call__(self, current_loss):
        if self.best_loss is None:
            self.best_loss = current_loss
        elif self.best_loss - current_loss > self.min_delta:
            self.best_loss = current_loss
            self.patience_counter = 0  # сброс счётчика терпения
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.patience:
            print("Early stopping triggered. Training stopped.")
            return True
        return False


class ModelTrainer:
    def __init__(self, model, model_name, save_path, loss_fn, optimizer, device, metrics=None, stoping: bool = False, lr_patience=5, lr_threshold=1e-3, stop_delta=0.5, stop_patience=10):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.metrics = metrics or {}
        self.history = {"train_loss": [], "valid_loss": []}
        for metric in self.metrics:
            self.history[f"train_{metric}"] = []
            self.history[f"valid_{metric}"] = []
        
        self.model_name = model_name
        self.save_path = save_path
        
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=lr_patience, threshold=lr_threshold)
        
        self.early_stopping = EarlyStopping(min_delta=stop_delta, patience=stop_patience) if stoping else None

        # Initialize the interactive mode for plotting
        plt.ion()
        self.fig, self.axs = (
            plt.subplots(1, 2, figsize=(8, 3))
            if self.metrics
            else plt.subplots(1, 1, figsize=(5, 3))
        )

    def fit(
        self,
        train_data,
        valid_data,
        batch_size,
        epochs,
        test_data=None,
    ):
        if isinstance(train_data, tuple):
            train_data = DataLoader(
                list(zip(train_data[0], train_data[1])),
                batch_size=batch_size,
                shuffle=True,
            )
        if isinstance(valid_data, tuple):
            valid_data = DataLoader(
                list(zip(valid_data[0], valid_data[1])), shuffle=False
            )

        best_valid_loss = float("inf")

        start_time_sec = time.time()

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")

            train_results = self.train_one_epoch(train_data)
            valid_results = self.validate_one_epoch(valid_data)
            self.update_history(train_results, valid_results)

            # Print progress
            self.print_epoch_status(train_results, valid_results)

            # Saving model if it's the best one
            if valid_results["loss"] < best_valid_loss:
                best_valid_loss = valid_results["loss"]
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.save_path, f"{self.model_name}_best.pth"),
                )
                torch.save(
                    self.model, os.path.join(self.save_path, f"{self.model_name}_full_best.pth")
                )

            # Updating the learning curves
            self.update_learning_curves()

            self.scheduler.step(valid_results["loss"])  # передаём значение потерь на валидации

            if self.early_stopping and self.early_stopping(valid_results["loss"]):
                break  # Выход из цикла, если сработала ранняя остановка

        end_time_sec = time.time()
        total_time_sec = end_time_sec - start_time_sec
        time_per_epoch_sec = total_time_sec / epochs
        print()
        print("Time total:     %5.2f sec" % (total_time_sec))
        print("Time per epoch: %5.2f sec" % (time_per_epoch_sec))

        # Save the training history to a file
        np.save(os.path.join(self.save_path, f"{self.model_name}_history.npy"), self.history)

        # Save the final plot to a file
        self.save_final_plot(self.save_path, self.model_name)
        plt.ioff()  # Turn off the interactive mode

        if test_data is not None:
            self.test(test_data)

    def train_one_epoch(self, data_loader):
        self.model.train()
        metrics = {metric: 0 for metric in self.metrics}
        total_loss = 0
        correct = 0
        total = 0
        progress_bar = tqdm(
            data_loader,
            desc="Training",
            ncols=60,
            leave=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
            # ascii="|", colour='g',
        )
        for x, y in progress_bar:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.loss_fn(outputs, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
            for metric in self.metrics:
                metrics[metric] += self.metrics[metric](outputs, y).item()
            # progress_bar.set_postfix(loss=loss.item())
        metrics = {metric: metrics[metric] / len(data_loader) for metric in metrics}
        metrics["loss"] = total_loss / len(data_loader)
        return metrics

    def validate_one_epoch(self, data_loader):
        self.model.eval()
        metrics = {metric: 0 for metric in self.metrics}
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                loss = self.loss_fn(outputs, y)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y).sum().item()
                total += y.size(0)
                for metric in self.metrics:
                    metrics[metric] += self.metrics[metric](outputs, y).item()
        metrics = {metric: metrics[metric] / len(data_loader) for metric in metrics}
        metrics["loss"] = total_loss / len(data_loader)
        return metrics

    def update_history(self, train_results, valid_results):
        for key in train_results:
            self.history[f"train_{key}"].append(train_results[key])
            self.history[f"valid_{key}"].append(valid_results[key])

    def print_epoch_status(self, train_results, valid_results):
        for key in train_results:
            print(
                f" - train_{key}: {train_results[key]:.4f} - val_{key}: {valid_results[key]:.4f}",
                end="",
            )
        print(f" - lr: {self.scheduler.optimizer.param_groups[0]['lr']}")

    def plot_learning_curves(self):
        fig, axs = (
            plt.subplots(1, 3, figsize=(15, 5))
            if self.metrics
            else plt.subplots(1, 2, figsize=(10, 5))
        )
        for i, key in enumerate(["loss"] + list(self.metrics.keys())):
            axs[i].plot(self.history[f"train_{key}"], label=f"Train {key.capitalize()}")
            axs[i].plot(self.history[f"valid_{key}"], label=f"Valid {key.capitalize()}")
            axs[i].set_title(f"{key.capitalize()} over Epochs")
            axs[i].legend()
        plt.show()

    def update_learning_curves(self):
        for i, key in enumerate(["loss"] + list(self.metrics.keys())):
            self.axs[i].clear()
            self.axs[i].plot(
                self.history[f"train_{key}"], label=f"Train {key.capitalize()}"
            )
            self.axs[i].plot(
                self.history[f"valid_{key}"], label=f"Valid {key.capitalize()}"
            )
            self.axs[i].set_title(
                f"{key.capitalize()} over Epochs",
            )
            self.axs[i].legend()

        plt.draw()
        plt.pause(0.001)  # Pause to ensure the plot updates

    def save_final_plot(self, save_path, model_name):
        plt.savefig(os.path.join(save_path, f"{model_name}_training_plot.png"))

    def test(self, test_data):
        if isinstance(test_data, tuple):
            test_data = DataLoader(list(zip(test_data[0], test_data[1])), shuffle=True)
        self.model.eval()
        metrics = {metric: 0 for metric in self.metrics}
        total_loss = 0
        with torch.no_grad():
            model = torch.load(os.path.join(self.save_path, f"{self.model_name}_full_best.pth"), weights_only=False).to(self.device)
            model.eval()
            for x, y in test_data:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                loss = self.loss_fn(outputs, y)
                total_loss += loss.item()
                outputs = self.model(x)
                loss = self.loss_fn(outputs, y)
                total_loss += loss.item()
                for metric in self.metrics:
                    metrics[metric] += self.metrics[metric](outputs, y).item()
        metrics["loss"] = total_loss
        for metric in metrics:
            metrics[metric] /= len(test_data)
        print("Test Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.2f}")
