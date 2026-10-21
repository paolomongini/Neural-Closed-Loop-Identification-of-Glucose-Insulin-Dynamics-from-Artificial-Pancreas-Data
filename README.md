# Neural Closed-Loop Identification of Glucose-Insulin Dynamics from Artificial Pancreas Data

Code for the identification of patient-specific glucose-insulin models from closed-loop
artificial pancreas data, as described in the paper of the same name.

The glucose model is a **DualSSM**, the difference of two deep state-space models (`neural_ssm`):

```
y_hat = SSM_0(meal) - F(SSM_1(insulin))
```

where `F` is a fourth-order low-pass filter. The data are generated in closed loop by a PID
controller; a differentiable model of the PID and of the insulin pump is used to simulate the
identified model in feedback.

## Pipeline

For each patient:


1. **Baseline (virtual patient 101).** The model is pre-trained on open-loop data of a reference
   virtual patient, normalized with the scalers of the target patient.
   * *Open loop*: SSM_0 is trained on a meal scenario, SSM_1 on an insulin scenario.
   * *Closed loop* (required by strategy 2 only): starting from the open-loop meal branch, the
     DualSSM is trained in feedback with the PID model on the insulin scenario.
2. **Fine-tuning on the target patient**, on 30 days of PID data (80% training, 10% validation,
   10% test, consecutive days, first 5 hours discarded), with one of two strategies:
   * **Strategy 1 (s1), open loop.** The DualSSM is trained on the recorded meal and insulin signals.
   * **Strategy 2 (s2), closed loop.** The DualSSM is trained in negative feedback with the PID model.

Both strategies minimize

```
MSE(y_hat, y) + prior_weight * L_prior + L_der2 + L_der3
```

* `L_prior`: physiological prior on the insulin effect. The insulin delivered over a sliding window
  is mapped through the correction factor of the patient into an expected glucose drop, which the
  insulin branch is required to follow.
* `L_der2`: L1 norm of the second differences of the two branch outputs.
* `L_der3`: maximum third difference of the responses to synthetic worst-case insulin and meal impulses.

The model with the lowest validation loss is retained.

## Repository structure

```
identification.ipynb      full pipeline for one patient; cells are selected by tag
run_identification.py     runs the notebook for several patients in parallel (papermill)
src/
  config.py               hyperparameters used in the paper
  dataset.py              data loading, normalization, synthetic impulses
  dual_ssm.py             DualSSM and construction of the deep SSMs
  models.py               physiological prior, PID controller model, closed-loop system
  parameters.py           PID, pump and patient parameters
  utils.py                low-pass filter, metrics, plotting, helpers
  neural_ssm_fastloop.py  faster step-by-step evaluation of neural_ssm (numerically equivalent)
```

## Installation

Python 3.11 was used.

```
pip install -r requirements.txt
```

`neural_ssm` must be version 0.28: `src/neural_ssm_fastloop.py` patches a method of this version.

## Data

Data and trained models are not distributed with this repository. The code expects the following
files, produced with the UVA/Padova simulator (one `.mat` file per virtual patient, sampled every minute):

```
data/
  CRtv.mat                                          time-varying carbohydrate ratio of all patients
  PID/train/sc_30days_identification/s#adult#XXX.mat   30 days of PID closed-loop data, patients 001-100
  train_101/30days_101_M/s#adult#101.mat            virtual patient 101, meal scenario
  train_101/30days_101_I_v2/s#adult#101.mat         virtual patient 101, insulin scenario
  test_ir/sc_1_day_test_IR_insulin_30g_13h/s#adult#XXX.mat   one-day insulin impulse test
  test_ir/sc_1_day_test_IR_meal_80g_13h/s#adult#XXX.mat      one-day meal impulse test
```

Each patient file contains the fields `injection`, `CGM`, `G`, `carb_intake` and
`basal_pattern_original`; the closed-loop files additionally contain the correction factor in
`iAP.RCM_param.CFpatientForModel`.

## Usage

Set the options at the top of `run_identification.py` and run it from the repository root:

```
python run_identification.py
```

| Option | Description |
|----|----|
| `STRATEGY` | `"s1"` (open-loop fine-tuning) or `"s2"` (closed-loop fine-tuning) |
| `EXP_NAME` | experiment folder under `models/`, `output/`, `results/` |
| `TRAIN_BASELINE` | train the baseline needed by the strategy (open loop for s1, open and closed loop for s2) |
| `TEST_ONLY` | evaluate the saved models without training |
| `PATIENTS` | list of patients (default 1-100) |
| `MAX_WORKERS` | patients processed in parallel (CPU) |

To reproduce the paper:

```
# 1) strategy 1, including the open-loop baseline
STRATEGY = "s1"; TRAIN_BASELINE = True

# 2) strategy 2, including the closed-loop baseline
STRATEGY = "s2"; TRAIN_BASELINE = True
```

The baseline is stored per patient in `models/baseline/` and shared by the two strategies.
The notebook can also be run interactively for a single patient by editing its `parameters` cell.

## Parameters used in the paper

All values are defined in `src/config.py`.

**Architecture** (identical for SSM_0 and SSM_1)

| Parameter | Value |
|----|----|
| layers | 4 |
| `d_model`, `d_state`, `d_hidden` | 8, 8, 8 |
| feed-forward | GLU |
| parametrization | `l2n` |
| `max_phase_b` | 0.04 |
| `dim_amp` | 3 |
| low-pass filter on the insulin branch | 4 coincident real poles, cut-off period 90 min |

**Training**

| Parameter | Baseline open loop | Baseline closed loop | Strategy 1 | Strategy 2 |
|----|----|----|----|----|
| epochs | 1500 | 1000 | 2000 | 2000 |
| optimizer | Adam, lr 1e-3 | Adam, lr 1e-3 | Adam, lr 1e-3 | Adam, lr 1e-3 |
| prior weight | - | - | `prior_weight_s1 = 1` | `prior_weight_s2 = 0.1` |
| prior scaling | - | - | `aggressive_coeff_s2 = 1` | `aggressive_coeff_s2 = 1` |
| prior on meal-free rollout | - | - | `correct_prior_for_s2 = True` | `correct_prior_for_s2 = True` |
| `loss_coeff_der` (second differences) | - | - | 1e-3 | 1e-3 |
| `loss_coeff_der3` (third differences) | - | - | 0.01 | 0.01 |

**Common settings**

| Parameter | Value |
|----|----|
| sampling time | 5 min |
| training sequences | 12 consecutive sequences of 48 h, one batch |
| prior window (`cumulative_window`) | 48 samples (4 h) |
| prior horizon (`horizon`) | 6 samples (30 min) |
| reference glucose of the prior | 110 mg/dL |
| closed-loop warm-up | first 10 epochs on the first 250 samples of each sequence |
| divergence handling | restart with a new seed (at most 100 restarts) |
| seed | 42 |

## Outputs

* `models/baseline/paz_XXX/{meal_s1, insulin_s1, closed_loop_s2}/`: baseline models.
* `models/<EXP_NAME>/paz_XXX/strategy_{1,2}/`: identified models (`trained_models.pth`) and loss histories (`training_log.mat`); `scalers.mat` contains the normalization bounds.
* `output/<EXP_NAME>/patient_XXX.ipynb`: executed notebooks, with training logs and plots.
* `results/<EXP_NAME>.csv`: one row per patient with
  * `FIT_train`: median FIT over the training sequences \[%\]
  * `FIT_test`: FIT on the test days \[%\]
  * `delta_G_insulin`, `delta_t_insulin`: glucose drop \[mg/dL\] and time to the minimum \[min\] in the insulin impulse test
  * `delta_G_meal`, `delta_t_meal`: glucose rise \[mg/dL\] and time to the maximum \[min\] in the meal impulse test

  The extremum is searched between 15 min and 6 h after the impulse, and `delta_t` is measured from the start of this window.

  The FIT is `100 (1 - ||y - y_hat|| / ||y - mean(y)||)`. In strategy 2 all metrics are computed on closed-loop rollouts with the PID model.


