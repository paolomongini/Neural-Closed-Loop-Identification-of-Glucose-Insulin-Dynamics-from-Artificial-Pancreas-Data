"""Loading of the closed-loop simulation data (UVA/Padova simulator, 5-min sampling)."""

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset

from parameters import (CR_TUNED, CR_TUNED_101, PID_CONVERSION_INDEX, PID_INTEGRAL_DURATION, PID_K_D, PID_K_I,
                        PID_K_P, PID_REF, PID_TS_CONTROLLER, PUMP_QUANTUM, PUMP_SATURATION_MAX, TS_MEASUREMENT,
                        load_CR_profile)


def _read_signals(loaded):
    """Insulin delivered by the pump [U], CGM [mg/dL] and carbohydrate intake per sample [mg]."""
    insulin = loaded["injection"]["signals"][0, 0]["values"][0, 0][1::5, 0] / 6000
    carbs = loaded["carb_intake"]["signals"][0, 0]["values"][0, 0][1:-1:2].flatten()
    meal = carbs.reshape(-1, 5).sum(axis=1)
    return insulin, meal


def _basal_at(basal_time, basal_values, ToD):
    indices = np.where(basal_time <= ToD)[0]
    return basal_values[indices[-1]] if len(indices) > 0 else basal_values[-1]


def _booster_d(glucose_PID, disable_booster_d_pos=True):
    """Derivative correction factors of the PID (single-sequence version)."""
    d1 = glucose_PID[-1] - glucose_PID[-2]
    d2 = glucose_PID[-1] - glucose_PID[-3]
    d3 = glucose_PID[-1] - glucose_PID[-4]

    booster_d_neg = 0 if (d1 < 0 and d2 < 0 and d3 < 0) else 1
    booster_d_pos = 1.1 if (d1 > 0 and d2 > 0 and d3 > 0) else 1
    if d1 > 10 and d2 > 20:
        booster_d_pos = 2.5
    if d1 > 15 and d2 > 30:
        booster_d_pos = 3
    if d1 > 20 and d2 > 40:
        booster_d_pos = 4
    if disable_booster_d_pos:
        booster_d_pos = 1
    return booster_d_neg, booster_d_pos


def _pump_saturation(amount, saturation_error):
    """Quantization and saturation of the insulin pump (single-sample version)."""
    bolus = PUMP_QUANTUM * round((amount + saturation_error) / PUMP_QUANTUM)
    bolus = min(bolus, PUMP_SATURATION_MAX)
    return bolus, amount + saturation_error - bolus


class MinMaxScalerTorch:
    """Min-max normalization to [0, 1]."""

    def __init__(self):
        self.params = {}
        self.device = "cpu"

    def to(self, device):
        self.device = device
        for key in self.params:
            self.params[key] = self.params[key].to(device)
        return self

    def compute_norm_indexes(self, x):
        self.params["low"] = torch.min(x)
        self.params["high"] = torch.max(x)

    def normalize(self, x):
        low, high = self.params["low"].to(self.device), self.params["high"].to(self.device)
        return (x - low) / (high - low)

    def denormalize(self, x_norm):
        low, high = self.params["low"].to(self.device), self.params["high"].to(self.device)
        return x_norm * (high - low) + low


class LoadData(Dataset):
    """Closed-loop PID data of one virtual patient.

    The min-max scalers are fitted on the first `train_size` samples. The PID is replayed
    on the recorded CGM to reconstruct the pump saturation error, which is used to
    initialize the controller state in closed-loop rollouts.

    Each item is (meal, insulin, CGM, time), all normalized except time.
    If `disable_booster_d_pos` is True, the PID replay does not amplify the derivative action on rising glucose.
    """

    def __init__(self, patient, data_path, train_size, disable_booster_d_pos=True):
        loaded = sio.loadmat(f"{data_path}/s#adult#{patient:03d}.mat")
        I_sat_real, MH_real = _read_signals(loaded)
        CGM_real = loaded["CGM"]["signals"][0, 0]["values"][0, 0][1::5].flatten()

        basal_values = loaded["basal_pattern_original"]["values"][0, 0].flatten()
        basal_time = loaded["basal_pattern_original"]["time"][0, 0].flatten()
        CR_values = load_CR_profile(patient)
        CR_tuned = CR_TUNED_101 if patient == 101 else CR_TUNED[patient - 1]

        # ----- replay of the PID controller on the recorded CGM -----
        saturation_error = np.zeros(len(CGM_real))
        for i in range(len(CGM_real)):
            ToD = i * 5 % 1440
            CR_index = np.where(np.arange(0, 289) <= round((ToD % 1) * 1440 / 5))[0][-1]
            CR_now = CR_values[0, CR_index] / np.max(CR_values)

            if i < 5:
                bolus = 0
            else:
                glucose_PID = CGM_real[0:i + 1]
                start_idx = max(0, i - PID_INTEGRAL_DURATION)
                error_capped = np.maximum(PID_REF - glucose_PID[start_idx:i + 1], PID_REF - 140)
                e_sum = np.sum(error_capped)

                K_p = PID_K_P / CR_now
                K_d = PID_K_D / PID_TS_CONTROLLER / CR_now
                K_i = PID_K_I * PID_TS_CONTROLLER / CR_now

                e = PID_REF - glucose_PID[i]
                e_m1 = PID_REF - glucose_PID[i - 1]
                booster_d_neg, booster_d_pos = _booster_d(glucose_PID, disable_booster_d_pos)
                basal_PID = K_p * e + K_d * (e - e_m1) * booster_d_pos + K_i * e_sum
                bolus = basal_PID * PID_CONVERSION_INDEX * booster_d_neg * CR_tuned

            I_rec = _basal_at(basal_time, basal_values, ToD) / 60 * TS_MEASUREMENT + bolus
            previous_error = saturation_error[i - 1] if i > 0 else 0
            _, saturation_error[i] = _pump_saturation(I_rec, previous_error)

        # ----- normalization -----
        self.scaler_meal = MinMaxScalerTorch()
        self.scaler_meal.compute_norm_indexes(torch.from_numpy(MH_real[:train_size]).float())
        self.scaler_insulin = MinMaxScalerTorch()
        self.scaler_insulin.compute_norm_indexes(torch.from_numpy(I_sat_real[:train_size]).float())
        self.scaler_glucose = MinMaxScalerTorch()
        self.scaler_glucose.compute_norm_indexes(torch.from_numpy(CGM_real[:train_size]).float())

        self.MH = self.scaler_meal.normalize(torch.from_numpy(MH_real).float())
        self.I_sat = self.scaler_insulin.normalize(torch.from_numpy(I_sat_real).float())
        self.CGM = self.scaler_glucose.normalize(torch.from_numpy(CGM_real).float())
        self.sat_e = self.scaler_insulin.normalize(torch.from_numpy(saturation_error).float())
        self.time = torch.arange(0, len(CGM_real))

        class BasalPattern:
            def __init__(self, time, values):
                self.time = time
                self.values = values

        self.basal_vec = BasalPattern(torch.from_numpy(basal_time).float(), torch.from_numpy(basal_values).float())

    def __len__(self):
        return len(self.CGM)

    def __getitem__(self, idx):
        return self.MH[idx].float(), self.I_sat[idx].float(), self.CGM[idx].float(), self.time[idx].float()


class SimpleLoadData(Dataset):
    """Open-loop data of the virtual patient 101, normalized with the scalers of the target patient.

    Each item is (meal, bolus insulin, glucose, time); the basal insulin is removed.
    """

    def __init__(self, data_path, patient, scaler_insulin, scaler_glucose, scaler_meal):
        loaded = sio.loadmat(f"{data_path}/s#adult#{patient:03d}.mat")
        I_ba_bo, MH = _read_signals(loaded)
        I_ba_bo = torch.from_numpy(I_ba_bo).float()

        basal_values = loaded["basal_pattern_original"]["values"][0, 0].flatten()
        basal_time = loaded["basal_pattern_original"]["time"][0, 0].flatten()
        I_bo = torch.zeros_like(I_ba_bo)
        for i in range(len(I_ba_bo)):
            I_bo[i] = I_ba_bo[i] - _basal_at(basal_time, basal_values, i * 5 % 1440) / 60 * 5

        G = loaded["G"]["signals"][0, 0]["values"][0, 0][1::5].flatten()
        self.i = scaler_insulin.normalize(I_bo)
        self.y = scaler_glucose.normalize(torch.from_numpy(G).float())
        self.m = scaler_meal.normalize(torch.from_numpy(MH).float())
        self.time = torch.arange(0, len(self.y))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.m[idx], self.i[idx], self.y[idx], self.time[idx]


def load_impulse_test(data_path, patient, scaler_meal, scaler_insulin, scaler_glucose, glucose_key="CGM"):
    """One-day impulse-response test (isolated insulin bolus or meal).

    Returns meal, insulin, glucose and time as tensors of shape (1, T, 1).
    """
    loaded = sio.loadmat(f"{data_path}/s#adult#{patient:03d}.mat")
    insulin, meal = _read_signals(loaded)
    glucose = loaded[glucose_key]["signals"][0, 0]["values"][0, 0][1::5].flatten()

    u0 = scaler_meal.normalize(torch.from_numpy(meal).float())
    u1 = scaler_insulin.normalize(torch.from_numpy(insulin).float())
    y = scaler_glucose.normalize(torch.from_numpy(glucose).float())
    time = torch.arange(0, len(y))
    return tuple(x[None, :, None] for x in (u0, u1, y, time))


class FakeScenarioDataset:
    """Synthetic worst-case impulses used to penalize oscillations of the identified model.

    The insulin impulse (worst_insulin, in U) is normalized with the insulin scaler; the
    meal impulse (worst_meal) is given directly in normalized units.
    """

    def __init__(self, scaler_meal, scaler_insulin, worst_insulin=8, worst_meal=0.4 * 6,
                 sim_time_I=5000, sim_time_M=1000, Ts=5, impulse_idx=350):
        N_I = sim_time_I // Ts
        N_M = sim_time_M

        u_ins = torch.zeros(N_I)
        u_ins[impulse_idx] = worst_insulin
        self.u1_fict = scaler_insulin.normalize(u_ins[None, :, None])
        self.u0_fict_I = scaler_meal.normalize(torch.zeros(N_I)[None, :, None])

        u_meal = torch.zeros(N_M)
        u_meal[impulse_idx] = worst_meal
        self.u0_fict_M = u_meal[None, :, None]
        self.u1_fict_M = torch.zeros(N_M)[None, :, None]

    @staticmethod
    def _max_third_difference(y):
        return torch.max(torch.abs(y[:, 3:, :] - 3 * y[:, 2:-1, :] + 3 * y[:, 1:-2, :] - y[:, :-3, :]))

    def compute_der3_loss(self, SSM_0, SSM_1, l_filter, coeff=1.0, skip_transient=200):
        """Open-loop model (strategy 1): y = SSM_0(meal) - filter(SSM_1(insulin))."""
        y0_I, _ = SSM_0(self.u0_fict_I)
        y1_I, _ = SSM_1(self.u1_fict)
        der3_I = self._max_third_difference((y0_I - l_filter(y1_I))[:, skip_transient:, :])

        y0_M, _ = SSM_0(self.u0_fict_M)
        y1_M, _ = SSM_1(self.u1_fict_M)
        der3_M = self._max_third_difference((y0_M - l_filter(y1_M))[:, skip_transient:, :])

        return coeff * (der3_I + der3_M)

    def compute_der3_loss_s2(self, Dual_SSM, coeff=1.0, skip_transient=200):
        """Same penalty evaluated on the DualSSM (strategy 2)."""
        y_fict_I, _, _ = Dual_SSM(self.u0_fict_I, self.u1_fict)
        der3_I = self._max_third_difference(y_fict_I[:, skip_transient:, :])

        y_fict_M, _, _ = Dual_SSM(self.u0_fict_M, self.u1_fict_M)
        der3_M = self._max_third_difference(y_fict_M[:, skip_transient:, :])

        return coeff * (der3_I + der3_M)
