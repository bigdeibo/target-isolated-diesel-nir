"""Training loops for the target-isolated model and deep reference methods."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch

from .models import SpectralResNetRegressor, build_reference_model


def train_resnet(X_train, y_train, X_validation, y_validation, X_test, seed, device, encoder_state, max_epochs=500, patience=50):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SpectralResNetRegressor().to(device)
    if encoder_state is not None:
        model.encoder.load_state_dict(encoder_state)
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_x = torch.from_numpy(X_train).to(device)
    train_y = torch.from_numpy(y_train).to(device)
    validation_x = torch.from_numpy(X_validation).to(device)
    validation_y = torch.from_numpy(y_validation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    best_loss = np.inf
    best_state = None
    best_epoch = -1
    stale = 0
    step = 0
    started = time.time()
    for epoch in range(max_epochs):
        model.train()
        permutation = torch.randperm(len(train_x), generator=generator, device=device)
        for begin in range(0, len(permutation), 32):
            indices = permutation[begin:begin + 32]
            learning_rate = 0.001 * 0.001 ** (step / 10000)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            step += 1
            optimizer.zero_grad(set_to_none=True)
            loss = ((model(train_x[indices]) - train_y[indices]) ** 2).mean()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation_loss = float(((model(validation_x) - validation_y) ** 2).mean())
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            stale = 0
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a validation state")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        validation_prediction = model(validation_x).cpu().numpy().ravel()
        test_prediction = model(torch.from_numpy(X_test).to(device)).cpu().numpy().ravel()
    return validation_prediction, test_prediction, {"best_epoch": best_epoch, "seconds": time.time() - started}


def train_reference_network(name, X_train, y_train, X_validation, y_validation, seed, max_epochs, patience, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_reference_model(name, torch.from_numpy(X_train[:2]), seed).to(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_x = torch.from_numpy(X_train).to(device)
    train_y = torch.from_numpy(y_train).to(device)
    has_validation = X_validation is not None
    if has_validation:
        validation_x = torch.from_numpy(X_validation).to(device)
        validation_y = torch.from_numpy(y_validation).to(device)
    if name == "ipa":
        initial_lr, batch_size, regularization, loss_name = 0.001, 16, 0.0095, "mae"
    else:
        initial_lr, batch_size, regularization, loss_name = 0.01, 32, 0.001, "mse"
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    def regularization_loss():
        return regularization * sum((module.weight ** 2).sum() for module in model.modules() if isinstance(module, torch.nn.Conv1d))

    def data_loss(prediction, truth):
        return (prediction - truth).abs().mean() if loss_name == "mae" else ((prediction - truth) ** 2).mean()

    def evaluate(X, y):
        model.eval()
        with torch.inference_mode():
            prediction = model(X)
            return float(data_loss(prediction, y) + regularization_loss())

    best_loss = np.inf
    best_state = None
    best_epoch = -1
    stale = 0
    step = 0
    started = time.time()
    for epoch in range(max_epochs):
        model.train()
        permutation = torch.randperm(len(train_x), generator=generator, device=device)
        total_data_loss = 0.0
        for begin in range(0, len(permutation), batch_size):
            indices = permutation[begin:begin + batch_size]
            learning_rate = initial_lr * 0.001 ** (step / 10000)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            step += 1
            optimizer.zero_grad(set_to_none=True)
            prediction = model(train_x[indices])
            loss = data_loss(prediction, train_y[indices]) + regularization_loss()
            loss.backward()
            optimizer.step()
            total_data_loss += float(data_loss(prediction, train_y[indices]).detach()) * len(indices)
        monitored = evaluate(validation_x, validation_y) if has_validation else total_data_loss / len(train_x) + float(regularization_loss().detach())
        if monitored < best_loss:
            best_loss = monitored
            best_epoch = epoch
            stale = 0
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        else:
            stale += 1
        if has_validation and stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"best_epoch": best_epoch, "best_loss": best_loss, "epochs_run": epoch + 1, "seconds": time.time() - started}


def predict_network(model, X, device):
    model.eval()
    with torch.inference_mode():
        return model(torch.from_numpy(X).to(device)).detach().cpu().numpy().ravel()
