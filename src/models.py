"""Physiological prior, differentiable PID controller and closed-loop model."""

import torch
import torch.nn as nn


def monotonic_gain_loss(y_hat, u, scaler_y, scaler_u, cumulative_window, horizon, CF, G_bar,
                        aggressive_coeff=1.0, l_filter=None):
    """Prior on the glucose-lowering effect of insulin.

    For each time step, the insulin delivered over the previous `cumulative_window` samples
    is converted into an expected glucose drop through the correction factor CF, and the
    deviation of the predicted glucose from G_bar is penalized for differing from it.

    Args:
        y_hat: (batch, time, 1) normalized glucose response attributed to insulin
        u: (batch, time, 1) normalized insulin
        CF: correction factor [mg/dL/U]
        G_bar: reference glucose [mg/dL]
        aggressive_coeff: scaling of the expected glucose drop
        l_filter: if given, the expected glucose drop is low-pass filtered
    Returns:
        scalar loss, normalized by the total insulin in the window
    """
    cumulative_window = int(cumulative_window)
    horizon = int(horizon)

    # insulin summed over a sliding window: (batch, time - cumulative_window - horizon + 1, 1)
    u_windows = u.unfold(dimension=1, size=cumulative_window, step=1).permute(0, 1, 3, 2)
    u_sum = u_windows[:, :-horizon, :, :].sum(dim=2)

    u_sum_denorm = scaler_u.denormalize(u_sum)
    delta_u_sum_denorm = u_sum_denorm - torch.min(u_sum_denorm)

    prior_delta_y = -CF * delta_u_sum_denorm
    if l_filter is not None:
        l_filter.reset(prior_delta_y[:, 0, :])
        prior_delta_y = l_filter(prior_delta_y)

    delta_y_denorm = scaler_y.denormalize(y_hat[:, cumulative_window:-horizon + 1, :]) - G_bar
    error = scaler_y.normalize(aggressive_coeff * prior_delta_y - delta_y_denorm)

    return torch.abs(error).sum() / (u_sum.sum() + 1e-8)


class NonLinearController(nn.Module):
    """Batched, differentiable model of the PID controller and of the insulin pump.

    Each call performs one control step: it receives the normalized CGM of the current
    sample and returns the normalized insulin delivered by the pump.
    If `disable_booster_d_pos` is True, the derivative action is not amplified on rising glucose.
    """

    def __init__(self, P, PF, basal_vec, scaler_glucose, scaler_insulin, batch_size=1, device="cpu",
                 disable_booster_d_pos=True):
        super().__init__()
        self.device = device
        self.disable_booster_d_pos = disable_booster_d_pos
        self.P = P
        self.PF = PF
        self.scaler_glucose = scaler_glucose
        self.scaler_insulin = scaler_insulin
        self.batch_size = batch_size

        self.register_buffer("basal_vec_time", basal_vec.time.clone().detach())
        self.register_buffer("basal_vec_values", basal_vec.values.clone().detach())
        self.register_buffer("glucose_PID", P.PID_par.ref * torch.ones((batch_size, P.PID_par.integral_duration + 1)))
        self.register_buffer("saturation_error", torch.zeros(batch_size, 1))
        self.register_buffer("_error_threshold", torch.tensor(P.PID_par.ref - 140, dtype=torch.float32))

    def to(self, device=None, *args, **kwargs):
        if device is not None:
            self.device = device
        return super().to(device, *args, **kwargs)

    def reset(self, saturation_error_init=None, glucose_PID_init=None):
        """Set the pump residual (batch, 1) [U] and the CGM history (batch, integral_duration + 1) [mg/dL]."""
        P = self.P
        batch_size = saturation_error_init.shape[0] if saturation_error_init is not None else self.batch_size
        if saturation_error_init is None:
            saturation_error_init = torch.zeros(batch_size, 1, device=self.device)
        if glucose_PID_init is None:
            glucose_PID_init = P.PID_par.ref * torch.ones((batch_size, P.PID_par.integral_duration + 1), device=self.device)
        self.saturation_error = saturation_error_init.clone()
        self.glucose_PID = glucose_PID_init.clone()
        self.batch_size = batch_size

    def forward(self, CGM_i, i):
        """
        Args:
            CGM_i: (batch,) or (batch, 1) normalized CGM at the current sample
            i: (batch, 1) sample index
        Returns:
            (batch, 1) normalized insulin delivered by the pump
        """
        P, PF = self.P, self.PF
        batch_size = CGM_i.shape[0]
        CGM_i = CGM_i.squeeze(1) if CGM_i.dim() == 2 else CGM_i
        CGM_i = self.scaler_glucose.denormalize(CGM_i)
        self.glucose_PID = torch.cat([self.glucose_PID[:, 1:], CGM_i.unsqueeze(1)], dim=1)

        ToD = (i * 5) % 1440
        bolus = torch.zeros(batch_size, 1, device=self.device)

        # the PID is active from the fifth sample on
        if (i >= 5).any():
            error_capped = torch.maximum(P.PID_par.ref - self.glucose_PID, self._error_threshold)
            error_capped = torch.clamp(error_capped, max=1e30)
            e_sum = torch.sum(error_capped, dim=1).unsqueeze(1)

            CR_index = ((ToD % 1) * 1440 / 5).long()
            CR_values = P.Patient_par.CR_values_not_norm.to(CR_index.device)
            CR_now = CR_values[0, CR_index] / torch.max(CR_values)

            K_p = P.PID_par.K_p / CR_now
            K_d = P.PID_par.K_d / P.PID_par.tsController / CR_now
            K_i = P.PID_par.K_i * P.PID_par.tsController / CR_now

            e = (P.PID_par.ref - self.glucose_PID[:, -1]).unsqueeze(1)
            e_m1 = (P.PID_par.ref - self.glucose_PID[:, -2]).unsqueeze(1)
            booster_d_neg, booster_d_pos = PF.function_booster_d(self.glucose_PID, self.disable_booster_d_pos)

            basal_PID = K_p * e + K_d * (e - e_m1) * booster_d_pos + K_i * e_sum
            CR_tuned = P.Patient_par.CR_tuned_101[0] if P.patient == 101 else P.Patient_par.CR_tuned[P.patient - 1]
            bolus = basal_PID * P.PID_par.conversion_index * booster_d_neg * CR_tuned

        basal = PF.calculate_basal(self.basal_vec_time, self.basal_vec_values, ToD).to(torch.float32)
        I = basal / 60 * P.PID_par.ts_measurement + bolus.detach().float()

        I_sat, _, self.saturation_error = PF.pump_saturation(
            I, torch.zeros(batch_size, 1, device=self.device), P.PID_par.ts_measurement, P.pumpParameter, self.saturation_error)

        return self.scaler_insulin.normalize(I_sat)


class ClosedLoopSystem_SSM(nn.Module):
    """Negative feedback interconnection of the DualSSM with the PID controller model.

    At each step the controller acts on the previous model output; its insulin is fed,
    with negative sign, to the insulin branch together with the external insulin input.
    """

    def __init__(self, system_model, controller):
        super().__init__()
        self.system_model = system_model
        self.controller = controller

    def reset(self, saturation_error_init, glucose_PID_init, u1_t0):
        self.controller.reset(saturation_error_init=saturation_error_init, glucose_PID_init=glucose_PID_init)
        self.system_model.reset(u1_t0[:, 0:1, :])
        self.system_model.reset_f()

    def forward(self, u_ext, time, y_t0, u1_t0):
        """
        Args:
            u_ext: (batch, horizon, 2) normalized meal and insulin inputs
            time: (batch, horizon, 1) sample indices
            y_t0: (batch, 1, 1) normalized initial glucose
            u1_t0: (batch, 1, 1) initial insulin, used to initialize the filter
        Returns:
            y, y0, y1: (batch, horizon, 1) trajectories [y(0), ..., y(T-1)] of the total
            output and of the two branches
        """
        batch_size = u_ext.shape[0]
        dim_in_0 = self.system_model.SSM_0.dim_in
        dim_in_1 = self.system_model.SSM_1.dim_in

        y = y_t0
        y0 = torch.zeros_like(y)
        y1 = -y
        y_prev = y

        self.system_model.reset(u1_t0[:, 0:1, :])
        self.system_model.reset_f()

        y_traj, y0_traj, y1_traj = [], [], []
        for i in range(time.shape[1]):
            y_traj.append(y)
            y0_traj.append(y0)
            y1_traj.append(y1)

            t_i = time[:, i:i + 1, :].view(batch_size, 1)
            control_u = -self.controller(y_prev.squeeze(dim=1), t_i).unsqueeze(dim=1)

            u0_t = u_ext[:, i:i + 1, :dim_in_0]
            u1_t = u_ext[:, i:i + 1, dim_in_0:dim_in_0 + dim_in_1] + control_u
            y, y0, y1 = self.system_model.forward(u0_t, u1_t)
            y_prev = y

        return torch.cat(y_traj, dim=1), torch.cat(y0_traj, dim=1), torch.cat(y1_traj, dim=1)
