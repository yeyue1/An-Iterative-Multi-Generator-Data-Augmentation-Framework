# Closed-loop multi-generator augmentation for PV fault diagnosis

This repository contains the implementation of the feedback-guided
multi-generator data-augmentation framework described in the manuscript
*An Iterative Multi-Generator Data Augmentation Framework for Photovoltaic
Fault Diagnosis*. It covers the closed-loop policy, its ablation variants, the
comparison baselines, and the additional analyses reported in the revised
manuscript.

The field I-V dataset is not included (commercial confidentiality, see the
manuscript's Data-availability statement). All scripts read a CSV or NPZ file
from `--data-path`, and a small synthetic demo file can be generated for smoke
tests.

## Environment

Python 3.10 with PyTorch, NumPy, scikit-learn, pandas, and Matplotlib:

```powershell
pip install -r requirements.txt
```

## Dataset format

CSV: a label column (`label` by default) plus one column per I-V point
(`f0` ... `f99`, or any names when `--feature-prefix` is used). Optional
metadata columns (`irradiance`, `temperature`, `module_id`, `condition_id`)
are excluded from the features and carried through the split.

NPZ: an array `x` of shape `[n_samples, 100]` and an array `y` with either the
string labels `NS`, `SC`, `OC`, `PS`, `DOA`, `DOA+SC`, `DOA+OC` or integer IDs
`0..6` in that order. Optional metadata arrays use the same column names.

## Experimental protocol

Three real-sample scenarios are constructed before augmentation (Table 2):
`balanced` (160 samples/class), `imbalanced` (all 6710 records), and `scarce`
(about 10% of each class, 671 records). Each scenario is split 60/20/20 into
training, validation, and test subsets with record-level stratification.

**Every run and every method uses one fixed partition per scenario**; only
model initialization and stochastic training differ between runs. The shared
partition is built from the data seed (`--partition-seed`, default 42; set it
explicitly to reproduce a partition), while run `i` uses model seed
`partition_seed + 100*i`. Validation data drive feedback, Gate 2, MOPSO
objectives, and stopping; the test subset is used only for final reporting.
All scripts follow this convention.

Unless noted, reported values are the mean and sample standard deviation
(`ddof=1`) over five matched runs. No p-values are computed: differences
between methods are descriptive matched-run differences.

## Paper-to-code map

| Manuscript item | Script | Output |
| --- | --- | --- |
| Table 4 (implementation settings) | `pv_fault_aug/config.py` | `config.json` written by each script |
| Table 5 (comparison with baselines) | `run_baselines.py --scenario all` | `comparison_table.csv`, `summary.json`, `runs.csv` |
| Table 6 (offline cost) | `run_baselines.py --scenario scarce` | `cost_table.csv` |
| Table 7 (component ablation) | `run_experiment.py --variant ...` | `{scenario}_summary.json` |
| Table 8 (per-class F1) | `run_baselines.py` / `run_analysis.py` | `per_class_summary.csv` |
| Table 9 (class-pair confusion) | `run_analysis.py` | `class_pairs.csv` |
| Table 10 (classifier generality) | `run_classifier_generality.py` | `classifier_generality.csv` |
| Table 11 (sensitivity and stopping rule) | `run_sensitivity.py` | `sensitivity.csv` |
| Table 12 (generator pool, natural mode) | `run_generator_pool.py --mode natural` | `pool_natural_summary.csv` |
| Table 13 (same-budget generator pool) | `run_generator_pool.py --mode fixed --budget 500` | `pool_fixed_summary.csv` |
| Table 14 (irradiance/temperature regimes) | `run_analysis.py` | `regimes.csv` |
| Table 15 (measurement/domain perturbations) | `run_analysis.py` | `perturbations.csv`, `group_heldout.csv` |
| Table 16 (physical I-V parameters) | `run_analysis.py` | `physical_parameters.csv` |
| Fig. 5 (run-level F1 and stopping) | `make_figures.py` | `f1_convergence.png` |
| Fig. 6 (quota distribution, scarce) | `make_figures.py` | `quota_distribution.png` |
| Fig. 7 (real vs generated) | `run_analysis.py` | `figures/real_vs_generated.png` |
| Fig. 8 (confusion matrices) | `run_analysis.py` | `figures/confusion_matrices.png` |

## Main commands

```powershell
# Demo smoke test (synthetic data, not the field data)
python run_experiment.py --data-path demo_data\pv_fault_demo.npz --make-demo-data
python run_experiment.py --data-path demo_data\pv_fault_demo.npz --scenario scarce --generators cvae --runs 1 --quick

# Proposed framework, one scenario
python run_experiment.py --data-path path\to\pv_fault_data.csv --scenario scarce --output-dir outputs\scarce

# Three scenarios for Table 5 and the ablation of Table 7
python run_baselines.py --data-path path\to\pv_fault_data.csv --scenario all --runs 5
python run_experiment.py --data-path path\to\pv_fault_data.csv --scenario balanced --variant no_feedback
python run_experiment.py --data-path path\to\pv_fault_data.csv --scenario imbalanced --variant no_difficulty
python run_experiment.py --data-path path\to\pv_fault_data.csv --scenario scarce --variant no_dual_gate
python run_experiment.py --data-path path\to\pv_fault_data.csv --scenario scarce --variant no_convergence

# Generator pool: natural (Table 12) and controlled same-budget (Table 13)
python run_generator_pool.py --data-path path\to\pv_fault_data.csv --scenario scarce --mode natural
python run_generator_pool.py --data-path path\to\pv_fault_data.csv --scenario scarce --mode fixed --budget 500

# Error analysis, robustness, and physical checks
python run_analysis.py --data-path path\to\pv_fault_data.csv --scenario scarce --runs 5

# Classifier generality (Table 10) and sensitivity (Table 11)
python run_classifier_generality.py --data-path path\to\pv_fault_data.csv --scenario scarce
python run_sensitivity.py --data-path path\to\pv_fault_data.csv --scenario scarce

# Figures 5 and 6 from the run directories written by run_experiment.py
python make_figures.py --runs-root outputs --output-dir outputs\figures
```

Use `--quick` for a fast syntax and pipeline check before the full run; the
full six-generator loop with the manuscript settings is computationally
expensive, especially the diffusion generators and the R_eval = 3 Gate-2
evaluations per class and round.

## Ablation variants

`--variant` implements the component-level ablation of Table 7:

- `baseline`: no augmentation (`max_iterations = 0`).
- `no_feedback`: class quotas and generator weights are fixed and uniform;
  MOPSO probing and contribution-score updates are skipped.
- `no_difficulty`: feedback still drives generator weighting and screening, but
  per-class quotas are uniform.
- `no_dual_gate`: Gate 1 and Gate 2 are disabled; the round-level F1-gain
  acceptance criterion of the convergence module is retained, so a round whose
  candidates deteriorate validation F1-macro is discarded.
- `no_convergence`: the loop runs for the full `max_iterations` budget without
  F1-saturation / feature-center stopping.

## Sensitivity options

`run_sensitivity.py` evaluates the eight rows of Table 11: the default
`(0.4, 0.3, 0.3)` difficulty weights, uniform weights, error-only,
scarcity-only, MOPSO without the entropy objective, either-condition stopping,
and joint stopping with `tau_stop = 0.005` and `0.020`. The same switches are
available directly on `run_experiment.py`
(`--alpha-error/--alpha-uncertainty/--alpha-scarcity`, `--stop-rule`,
`--stop-saturation`, `--stop-deterioration`, `--center-drift`,
`--no-entropy-objective`), so any other setting can be run through the main
entry point. The two `tau_stop` rows of Table 11 map to `--center-drift`.

## Outputs

`run_experiment.py` writes one directory per run with the final metrics,
per-class F1, the complete iteration history, the test predictions, and the
accepted synthetic samples. Summary files are written at the scenario level.
The other scripts write the CSV/JSON tables listed above plus `config.json`
recording the resolved configuration of the executed experiment.

## Modules

- `pv_fault_aug/data.py`: CSV/NPZ loading, scenario construction, fixed
  stratified split, training-set min-max normalization.
- `pv_fault_aug/models/generators.py`: WGAN-GP, CTGAN-style conditional GAN,
  CVAE, VAE-GAN, DDPM, and conditional DDPM.
- `pv_fault_aug/models/classifier.py`: 1D-CNN plus the 1D-ResNet and BiLSTM
  classifiers used in the generality analysis.
- `pv_fault_aug/feedback.py`: per-class error rate, entropy confidence, and
  scarcity.
- `pv_fault_aug/allocation.py`: difficulty weights, probing and operational
  quotas, generator-class ratios.
- `pv_fault_aug/mopso.py`: MOPSO with distributional, negative-F1, and entropy
  objectives and normalized Tchebycheff archive selection.
- `pv_fault_aug/screening.py`: class-wise MMD/Wasserstein Gate 1 and the paired
  R_eval class-bound marginal-gain Gate 2, plus the round-level F1 gain.
- `pv_fault_aug/pipeline.py`: the closed-loop process and result export.
- `pv_fault_aug/baselines.py`: no augmentation, SMOTE, one-shot generators,
  static ensemble, GAN+SMOTE, uncertainty-guided and curriculum-style
  baselines.
- `pv_fault_aug/analysis.py`: confusion/class-pair analysis, regime
  stratification, measurement and domain perturbations, group-held-out splits,
  and physical I-V parameters.
- `pv_fault_aug/stats.py`: paired differences and bootstrap intervals
  (descriptive only).
- `pv_fault_aug/figures.py`: run-level convergence, quota allocation,
  confusion-matrix, and three-panel real-versus-generated figures.

## Notes

The generated samples are clipped to the normalized `[0, 1]` I-V grid. Gate 2
and the round-level gain are paired: the R_eval baseline and temporary
classifiers share the same seed in each evaluation run. Physical parameters
(Voc, Isc, Pmp, fill factor) are computed for normal-condition samples, the
only class with a common reference operating point; pass `--voltage-max` and
`--current-max` to express them in volts, amperes, and watts, otherwise they
are reported on the normalized scale.