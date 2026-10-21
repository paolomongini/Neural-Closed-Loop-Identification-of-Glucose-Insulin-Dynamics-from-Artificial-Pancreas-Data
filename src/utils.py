import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Repository root: all data, model and result paths are resolved relative to it.
ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed=42):
    """Set the random seed of Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_3d(x):
    """Reshape a tensor to (batch, time, features)."""
    if x.ndim == 1:
        x = x.unsqueeze(0).unsqueeze(-1)
    elif x.ndim == 2:
        x = x.unsqueeze(-1)
    return x


class TrainingDivergence(RuntimeError):
    """Raised when a closed-loop rollout produces NaN or unbounded values."""


def check_divergence(y, threshold=1e4):
    if torch.isnan(y).any() or torch.max(torch.abs(y)) > threshold:
        raise TrainingDivergence("closed-loop rollout diverged")


def fun_start_controller(time_batch, loaded_parameters, scaler_glucose, scaler_insulin, dataset):
    """Initial state of the PID controller for rollouts starting at the first sample of each sequence.

    The state consists of the pump saturation error at the previous sample and of the
    CGM history over the integral window, both read from the recorded closed-loop data.

    Args:
        time_batch: sample indices, shape (batch, time) or (batch, time, 1).
    Returns:
        saturation_error_init: (batch, 1) [U]
        glucose_PID_init: (batch, integral_duration + 1) [mg/dL]
    """
    time_batch = time_batch.reshape(time_batch.shape[0], -1)
    previous_index = (time_batch[:, 0].int() - 1).unsqueeze(1)
    integral_window = previous_index + torch.arange(-loaded_parameters.PID_par.integral_duration, 1, device=previous_index.device)

    saturation_error_init = scaler_insulin.denormalize(dataset.sat_e[previous_index.long()].reshape_as(previous_index))
    glucose_PID_init = scaler_glucose.denormalize(dataset.CGM[integral_window.long()].reshape_as(integral_window))
    return saturation_error_init, glucose_PID_init


def second_difference_l1(y):
    """Sum of the absolute second differences along time."""
    return torch.sum(torch.abs(y[:, 2:, :] - 2 * y[:, 1:-1, :] + y[:, :-2, :]))


def FIT_formula(y_true, y_hat):
    """FIT = 100 * (1 - ||y - y_hat|| / ||y - mean(y)||), in percent."""
    numerator = np.linalg.norm(y_hat - y_true, ord=2)
    denominator = np.linalg.norm(y_true - np.mean(y_true), ord=2)
    return 100 * (1 - numerator / denominator)


def impulse_response_metrics(y_hat, u, scaler_glucose, mode, window=(3, 12 * 6)):
    """Peak glucose deviation after the largest input sample of a single impulse test.

    The extremum (minimum for insulin, maximum for meal) is searched between
    window[0] and window[1] samples after the impulse.

    Returns:
        delta_G: glucose deviation from the value preceding the impulse [mg/dL]
        delta_t: time from the start of the search window to the extremum [min]
    """
    k = torch.argmax(u[0, :, 0]).item()
    y_before = y_hat[0, k - 1, 0]
    y_window = y_hat[0, k + window[0]:k + window[1], 0]
    if mode == "insulin":
        y_peak, idx = torch.min(y_window, dim=0)
        delta_G = scaler_glucose.denormalize(y_before) - scaler_glucose.denormalize(y_peak)
    else:
        y_peak, idx = torch.max(y_window, dim=0)
        delta_G = scaler_glucose.denormalize(y_peak) - scaler_glucose.denormalize(y_before)
    return delta_G.item(), idx.item() * 5


def save_metrics(path, metrics):
    """Write a single-row CSV file with the given metrics."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def plot_glucose_insulin(time, insulin=None, meal=None, glucose=None, predicted_glucose=None, title=""):
    """Plot meal and glucose (top) and insulin (bottom)."""
    has_top = glucose is not None or predicted_glucose is not None or meal is not None
    n_plots = 2 if (insulin is not None and has_top) else 1
    if n_plots == 2:
        fig, axes = plt.subplots(2, 1, figsize=(10, 5.25), sharex=True, gridspec_kw={"height_ratios": [2, 1], "hspace": 0.05})
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 3.5))
        axes = [ax]

    ax_top = axes[0]
    if has_top:
        if meal is not None:
            mask = meal != 0
            ax_top.scatter(time[mask], meal[mask], color="#117733", s=60, alpha=0.8, label="Meal")
            ax_top.set_ylabel("Meal (normalized)")
        ax_glucose = ax_top.twinx()
        if predicted_glucose is not None:
            ax_glucose.plot(time, predicted_glucose, color="darkblue", linewidth=2, alpha=0.7, label="Predicted glucose")
        if glucose is not None:
            ax_glucose.plot(time, glucose, color="cornflowerblue", linewidth=2, alpha=0.7, label="Glucose")
        ax_glucose.set_ylabel("Glucose (mg/dL)")
        lines = ax_top.get_legend_handles_labels()[0] + ax_glucose.get_legend_handles_labels()[0]
        labels = ax_top.get_legend_handles_labels()[1] + ax_glucose.get_legend_handles_labels()[1]
        ax_glucose.legend(lines, labels, loc="upper right")
    ax_top.set_title(title)
    ax_top.grid(True, alpha=0.3)

    if insulin is not None:
        ax_ins = axes[-1]
        ax_ins.plot(time, insulin, color="tab:red", linewidth=2, label="Insulin")
        ax_ins.set_ylabel("Insulin (normalized)")
        ax_ins.grid(True, alpha=0.3)
        ax_ins.legend(loc="upper right")
    axes[-1].set_xlabel("Time step")
    fig.tight_layout()
    plt.show()


class LowPassFilter(nn.Module):
    """Discrete low-pass filter with `order` coincident real poles and unit static gain.

    Called on a sequence (batch, time, features), the filter is initialized at steady state
    on the first sample (or on `initial_value`) and run over the whole sequence.
    `forward` performs a single step on a sample (batch, features).
    """

    def __init__(self, T_sample=5, T_cut=90, order=4):
        super().__init__()
        f_sample = 1.0 / (T_sample * 60)
        f_cut = 1.0 / (T_cut * 60)
        p = np.exp(-2 * np.pi * f_cut / f_sample)

        a_poly = np.array([1.0])
        for _ in range(order):
            a_poly = np.convolve(a_poly, np.array([1.0, -p]))
        b_poly = np.zeros(order + 1)
        b_poly[0] = (1 - p) ** order

        self.order = order
        self.batch_size = None
        self.register_buffer("b", torch.tensor(b_poly, dtype=torch.float32))
        self.register_buffer("a_neg", torch.tensor(a_poly, dtype=torch.float32))

    def _init_buffers(self, batch_size, features, device):
        for name in ["x_hist", "y_hist"]:
            if name in self._buffers:
                del self._buffers[name]
            elif hasattr(self, name):
                delattr(self, name)
        self.register_buffer("x_hist", torch.zeros(batch_size, self.order, features, device=device))
        self.register_buffer("y_hist", torch.zeros(batch_size, self.order, features, device=device))
        self.batch_size = batch_size

    def reset(self, initial_value):
        """Fill input and output histories with `initial_value` of shape (batch, features)."""
        fill_value = initial_value[:, None, :].expand(-1, self.order, -1).contiguous()
        self.x_hist = fill_value.clone()
        self.y_hist = fill_value.clone()

    def forward(self, x_n):
        term_b = self.b[0] * x_n + torch.sum(self.b[None, 1:, None] * self.x_hist, dim=1)
        term_a = torch.sum(self.a_neg[None, 1:, None] * self.y_hist, dim=1)
        y_n = term_b - term_a
        self.x_hist = torch.cat([x_n[:, None, :], self.x_hist[:, :-1, :]], dim=1)
        self.y_hist = torch.cat([y_n[:, None, :], self.y_hist[:, :-1, :]], dim=1)
        return y_n

    def run(self, x_seq, initial_value=None):
        batch_size, time_steps, features = x_seq.shape
        if self.batch_size != batch_size:
            self._init_buffers(batch_size, features, x_seq.device)
        self.reset(x_seq[:, 0, :] if initial_value is None else initial_value[:, 0, :])

        y_total = torch.zeros_like(x_seq)
        for t in range(time_steps):
            y_total[:, t, :] = self.forward(x_seq[:, t, :])
        return y_total

    def __call__(self, x_seq, initial_value=None):
        return self.run(x_seq, initial_value)
