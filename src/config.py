"""Hyperparameters used for the results reported in the paper.

The same configuration is used for both identification strategies:
the keys ending in ``_s1`` are read only by strategy 1 (open-loop fine-tuning),
the keys ending in ``_s2`` only by strategy 2 (closed-loop fine-tuning).
"""

import copy

# Architecture of the two deep SSMs: index 0 is the meal branch, index 1 the insulin branch.
SSM_PARAM = {
    "input_dim": [1, 1],
    "output_dim": [1, 1],
    "d_model": [8, 8],
    "ff": ["GLU", "GLU"],
    "param": ["l2n", "l2n"],
    "d_state": [8, 8],
    "n_layers": [4, 4],
    "max_phase_b": 0.04,
    "d_hidden": [8, 8],
    "dim_amp": 3,
}

CONFIG = {
    # ----- optimization -----
    "learning_rate": 1e-3,
    "epochs_baseline_s1": 1500,     # open-loop pre-training on the virtual patient 101
    "epochs_baseline_s2": 1000,     # closed-loop pre-training on the virtual patient 101
    "epochs_s1": 2000,              # strategy 1 fine-tuning
    "epochs_s2": 1000,              # strategy 2 fine-tuning

    # ----- physiological prior on the insulin effect -----
    "cumulative_window": 12 * 4,    # samples of insulin summed by the prior (4 h)
    "horizon": 12 * 0.5,            # samples discarded at the end of the window (30 min)
    "prior_weight_s1": 1.0,
    "prior_weight_s2": 0.1,
    "aggressive_coeff_s2": 1.0,     # scaling of the prior glucose drop in strategy 2
    "correct_prior_for_s2": True,   # strategy 2: evaluate the prior on a meal-free closed-loop rollout

    # ----- smoothness regularization -----
    "loss_coeff_der": 1e-3,         # second-difference penalty on the two branch outputs
    "loss_coeff_der3": 0.01,        # third-difference penalty on synthetic worst-case impulse responses

    # ----- PID controller -----
    "disable_booster_d_pos": True,  # if True, the derivative action is not amplified when glucose is rising

    "SSM_param": SSM_PARAM,
}


def get_config():
    """Return an independent copy of the paper configuration."""
    return copy.deepcopy(CONFIG)
