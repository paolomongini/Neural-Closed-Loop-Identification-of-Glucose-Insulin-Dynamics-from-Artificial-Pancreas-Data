"""Faster step-by-step evaluation of neural_ssm==0.28 (Block2x2DenseL2SSM, mode="loop").

In the closed-loop rollouts of strategy 2 each SSM is called one time step at a time.
In the original implementation every call rebuilds the state-space matrices
(A_x, B_x, C_x, D_x) from the parameters, including a spectral normalization and a
linear solve, although they only depend on the parameters and are therefore constant
within a rollout.

This patch computes the matrices once per rollout and reuses them in the following
steps. The cache is rebuilt when a new rollout starts (state is None), when the
device or dtype changes, or when any parameter is modified in place (e.g. by an
optimizer step). The computations and the autograd graph are the same as in the
original implementation; the "scan" mode is left untouched.

Usage:
    import neural_ssm_fastloop
    neural_ssm_fastloop.apply()

Set Block2x2DenseL2SSM.FASTLOOP_ENABLED = False to fall back to the original code.
"""

import torch
from neural_ssm.ssm.lru import Block2x2DenseL2SSM

_ORIG_FORWARD = Block2x2DenseL2SSM.forward
_CACHE_ATTR = "_fastloop_cache"

Block2x2DenseL2SSM.FASTLOOP_ENABLED = True


def _version_key(self):
    # Tensor._version is incremented by every in-place update of the parameter.
    return (
        self.S._version,
        self.rho_raw._version,
        self.theta._version,
        self.K12_raw._version,
        self.K21_raw._version,
        self.K22_raw._version,
        self.log_gamma._version,
    )


def _build_xbasis_matrices(self, device, dtype):
    """Same computation as in the loop branch of Block2x2DenseL2SSM.forward."""
    A_z, B_z, C_z, D_z, _ = self.compute_z_matrices()

    S = self.S.to(device=device, dtype=dtype)
    I = torch.eye(self.d_state, device=device, dtype=dtype)
    ST_inv = torch.linalg.solve(S.T, I)

    A_x = S.T @ A_z.T @ ST_inv
    B_x = B_z.T @ ST_inv
    C_x = S.T @ C_z.T
    D_x = D_z.T
    return A_x, B_x, C_x, D_x


def _fastloop_forward(self, u, state=None, *, time_first=False, return_state=True, mode="scan", return_last=False):
    if mode == "scan" or not Block2x2DenseL2SSM.FASTLOOP_ENABLED:
        return _ORIG_FORWARD(self, u, state, time_first=time_first,
                             return_state=return_state, mode=mode, return_last=return_last)

    if u.dim() == 2:
        u = u.unsqueeze(0) if not time_first else u.unsqueeze(1)
    u_bt = u.transpose(0, 1).contiguous() if time_first else u
    B_sz, L, du = u_bt.shape
    assert du == self.d_input
    dx = self.d_state
    device, dtype = u_bt.device, u_bt.dtype

    if state is None:
        x0 = u_bt.new_zeros(B_sz, dx)
    else:
        x0 = state.unsqueeze(0).expand(B_sz, -1) if state.dim() == 1 else state
        x0 = x0.to(device=device, dtype=dtype)

    vkey = _version_key(self)
    cache = getattr(self, _CACHE_ATTR, None)
    rebuild = (cache is None or state is None or cache["device"] != device
               or cache["dtype"] != dtype or cache["vkey"] != vkey)
    if rebuild:
        A_x, B_x, C_x, D_x = _build_xbasis_matrices(self, device, dtype)
        cache = {"vkey": vkey, "device": device, "dtype": dtype, "A_x": A_x, "B_x": B_x, "C_x": C_x, "D_x": D_x}
        setattr(self, _CACHE_ATTR, cache)
    else:
        A_x, B_x, C_x, D_x = cache["A_x"], cache["B_x"], cache["C_x"], cache["D_x"]

    y_seq = u_bt.new_empty(B_sz, L, self.d_output)
    x_seq = u_bt.new_empty(B_sz, L + 1, dx) if return_state else None

    x = x0
    if return_state:
        x_seq[:, 0, :] = x
    for t in range(L):
        u_t = u_bt[:, t, :]
        y_seq[:, t, :] = x @ C_x + u_t @ D_x
        x = x @ A_x + u_t @ B_x
        if return_state:
            x_seq[:, t + 1, :] = x

    if time_first:
        y_seq = y_seq.transpose(0, 1).contiguous()
        if return_state:
            x_seq = x_seq.transpose(0, 1).contiguous()

    if return_state and return_last:
        return y_seq, x_seq, x
    if return_state:
        return y_seq, x_seq
    if return_last:
        return y_seq, x
    return y_seq


_fastloop_forward._is_fastloop_patch = True


def apply():
    """Install the patch (idempotent)."""
    if not getattr(Block2x2DenseL2SSM.forward, "_is_fastloop_patch", False):
        Block2x2DenseL2SSM.forward = _fastloop_forward
