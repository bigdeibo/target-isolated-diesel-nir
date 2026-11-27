"""PLS, SVR, and BOSS-based PLS reference methods."""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR

from .data import scale_xy
from .metrics import rmsep

PREPROCESSING = ("center", "snv", "sg1", "sg1_snv")


def preprocess(X_train, X_test, name):
    if name == "center":
        return X_train.copy(), X_test.copy()
    if name in ("snv", "sg1_snv"):
        if name == "sg1_snv":
            X_train = savgol_filter(X_train, 11, 2, deriv=1, axis=1)
            X_test = savgol_filter(X_test, 11, 2, deriv=1, axis=1)
        train_scale = X_train.std(1, keepdims=True)
        test_scale = X_test.std(1, keepdims=True)
        train_scale[train_scale == 0] = 1
        test_scale[test_scale == 0] = 1
        return (
            (X_train - X_train.mean(1, keepdims=True)) / train_scale,
            (X_test - X_test.mean(1, keepdims=True)) / test_scale,
        )
    if name == "sg1":
        return savgol_filter(X_train, 11, 2, deriv=1, axis=1), savgol_filter(X_test, 11, 2, deriv=1, axis=1)
    raise ValueError(name)


def fit_pls(X_train, y_train, X_test, preprocessing, components):
    A, B = preprocess(X_train, X_test, preprocessing)
    model = PLSRegression(n_components=components, scale=False, max_iter=500)
    model.fit(A, y_train)
    return model.predict(B).ravel()


def fit_svr(X_train, y_train, X_test, preprocessing, C, epsilon):
    A0, B0 = preprocess(X_train, X_test, preprocessing)
    A, a, B, y_mean, y_scale = scale_xy(A0, y_train, B0)
    model = SVR(C=C, epsilon=epsilon, gamma="scale", kernel="rbf")
    model.fit(A[:, 0, :], a.ravel())
    return model.predict(B[:, 0, :]) * y_scale + y_mean


def pls_predict(X_train, y_train, X_test, variables, components):
    variables = np.asarray(variables, dtype=int)
    model = PLSRegression(n_components=int(components), scale=False, max_iter=500)
    model.fit(X_train[:, variables], y_train)
    return model.predict(X_test[:, variables]).ravel()


def select_pls_components(X, y, variables, folds, max_components=15):
    variables = np.asarray(variables, dtype=int)
    upper = min(max_components, len(variables), len(y) - 1)
    rows = []
    for components in range(1, upper + 1):
        errors = []
        for train, validation in folds:
            prediction = pls_predict(X[train], y[train], X[validation], variables, components)
            errors.append(rmsep(y[validation], prediction))
        rows.append({"components": components, "cv_rmsep": float(np.mean(errors))})
    return min(rows, key=lambda row: (row["cv_rmsep"], row["components"]))


def boss_style_select(X, y, seed, folds, n_iter=20, shrink=0.8, min_vars=10, max_components=15):
    rng = np.random.default_rng(seed)
    active = np.arange(X.shape[1], dtype=int)
    best_full = select_pls_components(X, y, active, folds, max_components)
    best = {"variables": active.copy(), **best_full, "iteration": 0}
    history = [{"iteration": 0, "n_variables": len(active), "selected_components": best_full["components"], "best_cv_rmsep": best_full["cv_rmsep"]}]
    model = PLSRegression(n_components=best_full["components"], scale=False, max_iter=500)
    model.fit(X, y)
    weights = np.abs(model.coef_.ravel())
    weights /= max(weights.max(), 1e-12)
    for iteration in range(1, n_iter + 1):
        target = max(min_vars, int(np.ceil(len(active) * shrink)))
        probabilities = np.maximum(weights[active], 0) + 1e-12
        probabilities /= probabilities.sum()
        subset = np.sort(rng.choice(active, size=target, replace=False, p=probabilities))
        selected = select_pls_components(X, y, subset, folds, max_components)
        is_better = selected["cv_rmsep"] < best["cv_rmsep"] - 1e-12 or (
            np.isclose(selected["cv_rmsep"], best["cv_rmsep"], rtol=0, atol=1e-12)
            and len(subset) < len(best["variables"])
        )
        if is_better:
            best = {"variables": subset.copy(), **selected, "iteration": iteration}
        history.append({"iteration": iteration, "n_variables": len(subset), "selected_components": selected["components"], "best_cv_rmsep": selected["cv_rmsep"]})
        model = PLSRegression(n_components=selected["components"], scale=False, max_iter=500)
        model.fit(X[:, subset], y)
        coefficients = np.abs(model.coef_.ravel())
        coefficients /= max(coefficients.max(), 1e-12)
        weights = np.full(X.shape[1], 1e-12)
        weights[subset] = coefficients
        active = subset
        if len(active) == min_vars:
            break
    return best, history
