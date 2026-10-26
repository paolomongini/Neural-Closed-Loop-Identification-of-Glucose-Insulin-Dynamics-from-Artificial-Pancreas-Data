"""Parameters of the PID controller, of the insulin pump and of the virtual patients."""

import scipy.io as sio
import torch

from utils import ROOT

# ----- PID controller -----
PID_K_P = -0.0665
PID_K_I = -1.9342e-4
PID_K_D = -2.0922
PID_TS_CONTROLLER = 5.0         # [min]
PID_REF = 110                   # glucose set-point [mg/dL]
PID_CONVERSION_INDEX = 0.007
PID_INTEGRAL_DURATION = 12 * 2  # integral window [samples]
TS_MEASUREMENT = 5.0            # [min]

# ----- insulin pump -----
PUMP_QUANTUM = 0.05             # [U]
PUMP_SATURATION_MAX = 12.0      # [U]

# Patient-specific gain of the PID controller (patients 1-100 and the virtual patient 101).
CR_TUNED = [
    0.5, 1.8, 1.3, 4,   1,   0.8, 0.5, 1.2, 0.9, 2.2,   # 1-10
    1.2, 2.0, 1.6, 1.2, 0.9, 0.9, 1.2, 1.2, 1.2, 0.9,   # 11-20
    1,   0.7, 1.2, 1.2, 1.2, 1.2, 0.9, 1.2, 1.0, 1.2,   # 21-30
    1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.6, 1.2, 1.2, 1.2,   # 31-40
    0.9, 1.2, 1.2, 1.3, 1.7, 1.2, 2.0, 1.2, 0.9, 2.4,   # 41-50
    1.2, 0.8, 1.2, 1.2, 1.2, 0.7, 1.6, 2.4, 1.2, 1.2,   # 51-60
    1.8, 1.2, 2.2, 1.2, 0.6, 0.9, 1.5, 1.3, 1.5, 0.8,   # 61-70
    2.5, 1.4, 1.5, 1.2, 1.2, 1.2, 0.7, 1.4, 1.4, 2.3,   # 71-80
    2.0, 1.2, 1.2, 1.1, 1.2, 1.4, 1.3, 1.3, 1.2, 1.2,   # 81-90
    1.3, 1.3, 1.4, 1.5, 1.2, 1.9, 1.3, 1.5, 1.8, 1.2,   # 91-100
]
CR_TUNED_101 = 2.0


def load_CR_profile(patient):
    """Time-varying carbohydrate ratio of a patient, shape (1, 289) [g/U]."""
    return sio.loadmat(ROOT / "data" / "CRtv.mat")["CRtv"]["values"][0, patient - 1]


class Parameter:
    """Container of PID, pump and patient parameters used by the controller model."""

    def __init__(self, patient, data_path=None, device="cpu"):
        self.device = device
        self.patient = patient
        self.PID_par = self.PID_par()
        self.pumpParameter = self.pumpParameter(device)
        self.Patient_par = self.Patient_par(patient, data_path, device)

    def to(self, device):
        self.device = device
        self.Patient_par.to(device)
        self.pumpParameter.to(device)
        return self

    class PID_par:
        def __init__(self):
            self.K_p = PID_K_P
            self.K_i = PID_K_I
            self.K_d = PID_K_D
            self.tsController = PID_TS_CONTROLLER
            self.ref = PID_REF
            self.conversion_index = PID_CONVERSION_INDEX
            self.integral_duration = PID_INTEGRAL_DURATION
            self.ts_measurement = TS_MEASUREMENT

    class Patient_par:
        def __init__(self, patient, data_path, device):
            self.CR_tuned = torch.tensor(CR_TUNED, dtype=torch.float64).to(device)
            self.CR_tuned_101 = torch.tensor([CR_TUNED_101], dtype=torch.float64).to(device)
            self.CR_values_not_norm = torch.from_numpy(load_CR_profile(patient)).float().to(device)

            if patient != 101:
                # correction factor [mg/dL/U] stored with the closed-loop data of the patient
                iAP_data = sio.loadmat(f"{data_path}/s#adult#{patient:03d}.mat")
                self.CF = torch.from_numpy(iAP_data["iAP"]["RCM_param"][0, 0]["CFpatientForModel"][0, 0][0]).float().to(device)
            else:
                self.CF = torch.tensor([40]).to(device)

        def to(self, device):
            for name, value in vars(self).items():
                if isinstance(value, torch.Tensor):
                    setattr(self, name, value.to(device))
            return self

    class pumpParameter:
        def __init__(self, device):
            self.quantum = torch.tensor([PUMP_QUANTUM]).to(device)
            self.saturationMax = torch.tensor([PUMP_SATURATION_MAX]).to(device)

        def to(self, device):
            for name, value in vars(self).items():
                if isinstance(value, torch.Tensor):
                    setattr(self, name, value.to(device))
            return self


class StraightThroughRound(torch.autograd.Function):
    """Rounding in the forward pass, identity in the backward pass."""

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad):
        return grad


class StraightThroughClamp(torch.autograd.Function):
    """Clamping in the forward pass, identity in the backward pass."""

    @staticmethod
    def forward(ctx, x, min_val, max_val):
        return torch.clamp(x, min_val, max_val)

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None


class PID_functions:
    """Batched building blocks of the PID controller."""

    @staticmethod
    def pump_saturation(bolus, basal, ts_measurement, pumpParameter, saturation_error):
        """Quantization and saturation of the insulin pump.

        The requested amount is rounded to the pump quantum, accounting for the residual
        of the previous steps, and clipped to [0, saturationMax].

        Returns:
            bolus: delivered insulin [U], (batch, 1)
            basal: unchanged input
            saturation_error: residual carried to the next step [U], (batch, 1)
        """
        amount = bolus + basal / 60 * ts_measurement
        quantum = pumpParameter.quantum

        if torch.all(quantum == 0):
            bolus = amount
        else:
            bolus = quantum * StraightThroughRound.apply((amount + saturation_error) / quantum)

        bolus = StraightThroughClamp.apply(bolus, torch.tensor(0.0, device=bolus.device), pumpParameter.saturationMax)
        saturation_error = (amount + saturation_error - bolus).to(torch.float32).detach()
        return bolus.to(torch.float32), basal, saturation_error

    @staticmethod
    def function_booster_d(glucose_PID, disable_booster_d_pos=True):
        """Derivative correction factors of the PID, computed on the last four CGM samples.

        Args:
            glucose_PID: (batch, n) with n >= 4
            disable_booster_d_pos: if True, booster_d_pos is always 1
        Returns:
            booster_d_neg, booster_d_pos: (batch, 1)
        """
        d1 = glucose_PID[:, -1] - glucose_PID[:, -2]
        d2 = glucose_PID[:, -1] - glucose_PID[:, -3]
        d3 = glucose_PID[:, -1] - glucose_PID[:, -4]

        # insulin suspended when glucose is steadily decreasing
        booster_d_neg = torch.ones_like(d1)
        booster_d_neg[(d1 < 0) & (d2 < 0) & (d3 < 0)] = 0

        # derivative action amplified when glucose is rising
        booster_d_pos = torch.ones_like(d1)
        booster_d_pos[(d1 > 0) & (d2 > 0) & (d3 > 0)] = 1.1
        booster_d_pos = torch.where((d1 > 10) & (d2 > 20), torch.tensor(2.5, device=glucose_PID.device), booster_d_pos)
        booster_d_pos = torch.where((d1 > 15) & (d2 > 30), torch.tensor(3.0, device=glucose_PID.device), booster_d_pos)
        booster_d_pos = torch.where((d1 > 20) & (d2 > 40), torch.tensor(4.0, device=glucose_PID.device), booster_d_pos)
        if disable_booster_d_pos:
            booster_d_pos = torch.ones_like(d1)

        return booster_d_neg.unsqueeze(1), booster_d_pos.unsqueeze(1)

    @staticmethod
    def calculate_basal(basal_time, basal_values, ToD):
        """Basal rate active at the given time of day.

        Args:
            basal_time: (n_basal,) start times of the basal segments [min]
            basal_values: (n_basal,) basal rates [U/h]
            ToD: (batch, 1) time of day [min]
        Returns:
            (batch, 1) basal rate [U/h]
        """
        batch = ToD.shape[0]
        n_basal = basal_time.shape[0]
        mask = basal_time.unsqueeze(0).repeat(batch, 1) <= ToD.repeat(1, n_basal)
        last_idx = n_basal - 1 - torch.flip(mask, dims=[1]).float().argmax(dim=1)
        return basal_values[last_idx].unsqueeze(1)
