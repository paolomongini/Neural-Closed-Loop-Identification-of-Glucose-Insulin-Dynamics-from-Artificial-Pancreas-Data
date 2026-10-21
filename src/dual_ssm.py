import torch
import torch.nn as nn
from neural_ssm import DeepSSM


def build_deep_ssm(SSM_param, idx=0):
    """Deep SSM with the architecture of branch `idx` (0: meal, 1: insulin) of SSM_param."""
    return DeepSSM(d_input=SSM_param["input_dim"][idx], d_output=SSM_param["output_dim"][idx],
                   d_model=SSM_param["d_model"][idx], d_state=SSM_param["d_state"][idx],
                   n_layers=SSM_param["n_layers"][idx], ff=SSM_param["ff"][idx], param=SSM_param["param"][idx],
                   max_phase_b=SSM_param["max_phase_b"], d_hidden=SSM_param["d_hidden"][idx],
                   dim_amp=SSM_param["dim_amp"])


class DualSSM(nn.Module):
    """Glucose model with two additive branches:

        y_hat = SSM_0(meal) - filter(SSM_1(insulin))

    Calling the module processes whole sequences; `forward` performs a single time step
    and keeps the internal states of both SSMs and of the filter.
    """

    def __init__(self, SSM_0, SSM_1, filter, use_low_pass_I=True):
        super().__init__()
        self.SSM_0 = SSM_0
        self.SSM_1 = SSM_1
        self.filter = filter
        self.use_low_pass_I = use_low_pass_I

        self.SSM_0_x = None
        self.SSM_1_x = None
        self.SSM_0.dim_in = SSM_0.d_input
        self.SSM_1.dim_in = SSM_1.d_input
        self._y1_init = None

    def forward(self, u0_in, u1_in):
        """Single time step. u0_in, u1_in: (batch, 1, dim_in). Returns y_hat, y0, y1: (batch, 1, dim_out)."""
        y0, self.SSM_0_x = self.SSM_0.forward(u=u0_in, state=self.SSM_0_x, mode="loop")
        y1, self.SSM_1_x = self.SSM_1.forward(u=u1_in, state=self.SSM_1_x, mode="loop")
        if self.use_low_pass_I:
            y1 = self.filter.forward(y1[:, 0, :])[:, None, :]
        return y0 - y1, y0, y1

    def run(self, u0_seq, u1_seq):
        """Whole sequences. u0_seq, u1_seq: (batch, time, dim_in)."""
        y_SSM_0, _ = self.SSM_0(u0_seq)
        y_SSM_1, _ = self.SSM_1(u1_seq)
        if self.use_low_pass_I:
            y_SSM_1 = self.filter(y_SSM_1)
        return y_SSM_0 - y_SSM_1, y_SSM_0, y_SSM_1

    def __call__(self, u0_seq, u1_seq):
        return self.run(u0_seq, u1_seq)

    def reset(self, u1_0=None):
        """Reset the SSM states. If u1_0 (batch, 1, dim) is given, the insulin-branch output
        on it is stored to initialize the filter in `reset_f`."""
        self.SSM_0_x = None
        self.SSM_1_x = None
        if u1_0 is not None:
            with torch.no_grad():
                y1_0, _ = self.SSM_1.forward(u=u1_0, state=None, mode="loop")
            self._y1_init = y1_0

    def reset_f(self, u=None):
        """Initialize the filter at steady state on u[:, 0, :], or on the output stored by `reset`."""
        if u is None:
            u = self._y1_init
        batch_size, _, features = u.shape
        if self.filter.batch_size != batch_size:
            self.filter._init_buffers(batch_size, features, u.device)
        self.filter.reset(u[:, 0, :])
