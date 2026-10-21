"""Run the identification notebook for several patients in parallel with papermill.

The cells of identification.ipynb are selected by tag according to the settings below;
one executed notebook per patient is stored in output/<EXP_NAME>/, and the per-patient
metrics are collected in results/<EXP_NAME>.csv.
"""

import csv
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import nbformat
import papermill as pm
from nbformat.v4 import new_notebook

# ----------------------------- settings -----------------------------
STRATEGY = "s1"            # "s1": open-loop fine-tuning, "s2": closed-loop fine-tuning
EXP_NAME = STRATEGY        # experiment folder under models/, output/ and results/
TRAIN_BASELINE = True      # train the virtual-patient-101 baseline required by STRATEGY
TEST_ONLY = False          # evaluate previously saved models, without training
PATIENTS = list(range(1, 101))

MAX_WORKERS = 25           # patients processed in parallel
THREADS_PER_WORKER = 1     # the tensors are small: one thread per process is the most efficient choice
KERNEL_NAME = "python3"
# --------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "identification.ipynb"

# Inherited by every papermill kernel.
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = str(THREADS_PER_WORKER)


def tags_to_run():
    tags = ["parameters", "imports", "setup", "data", "helpers"]
    if TRAIN_BASELINE and not TEST_ONLY:
        tags += ["baseline_data", "baseline_s1"]
        if STRATEGY == "s2":
            tags += ["baseline_s2"]
    if not TEST_ONLY:
        tags += [f"{STRATEGY}_train"]
    tags += [f"{STRATEGY}_test"]
    return tags


def build_notebook(path, tags):
    """Write a notebook containing the cells of NOTEBOOK with the given tags, in the given order."""
    source = nbformat.read(NOTEBOOK, as_version=4)
    nb = new_notebook(metadata=source.metadata)
    for tag in tags:
        nb.cells += [c for c in source.cells if tag in c.metadata.get("tags", [])]
    nbformat.write(nb, path)


def run_patient(patient, tags):
    out_dir = ROOT / "output" / EXP_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    out_nb = out_dir / f"patient_{patient:03d}.ipynb"

    with tempfile.NamedTemporaryFile(suffix=".ipynb", delete=False) as tf:
        temp_nb = tf.name
    try:
        build_notebook(temp_nb, tags)
        print(f"[start] patient {patient}")
        pm.execute_notebook(temp_nb, str(out_nb), kernel_name=KERNEL_NAME, cwd=str(ROOT),
                            parameters={"patient": patient, "strategy": STRATEGY, "exp_name": EXP_NAME})
        print(f"[done]  patient {patient}")
    finally:
        os.remove(temp_nb)
    return out_nb


def collect_results():
    rows = []
    for f in sorted((ROOT / "results" / EXP_NAME).glob("paz_*.csv")):
        with open(f, newline="") as fh:
            rows += list(csv.DictReader(fh))
    if not rows:
        return
    out = ROOT / "results" / f"{EXP_NAME}.csv"
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"metrics of {len(rows)} patients written to {out}")


if __name__ == "__main__":
    tags = tags_to_run()
    print(f"strategy {STRATEGY}, experiment {EXP_NAME}, cells: {tags}")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(run_patient, p, tags): p for p in PATIENTS}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print(f"[error] patient {futures[fut]}: {e}")
    collect_results()
