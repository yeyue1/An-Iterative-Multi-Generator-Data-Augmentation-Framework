# PV closed-loop augmentation code

This folder contains the Python implementation corresponding to the paper draft.
The code is written so the real I-V dataset can be added later without changing
the closed-loop framework.

## Expected dataset format

CSV:

```text
label,f0,f1,...,f99
NS, ...
SC, ...
```

NPZ:

```python
np.savez("pv_fault_data.npz", x=x_array, y=labels)
```

`x` must have shape `[n_samples, 100]`. Labels can be the strings
`NS`, `SC`, `OC`, `PS`, `DOA`, `DOA+SC`, `DOA+OC`, or integer IDs
`0..6` in that order.

## Main command

From the repository root:

```powershell
python run_experiment.py --data-path path\to\pv_fault_data.csv --scenario scarce --output-dir outputs\scarce
```

If the real field dataset is not available yet, create a small demo NPZ file
and run a smoke test:

```powershell
python run_experiment.py --data-path demo_data\pv_fault_demo.npz --make-demo-data
python run_experiment.py --data-path demo_data\pv_fault_demo.npz --scenario scarce --generators cvae --runs 1 --quick
```

If the CSV uses different names, pass them explicitly:

```powershell
python run_experiment.py --data-path path\to\pv_fault_data.csv --label-column fault_type --feature-prefix iv_ --feature-dim 100
```

Use `--quick` for a syntax and pipeline smoke test before running full
experiments:

```powershell
python run_experiment.py --data-path path\to\pv_fault_data.csv --scenario scarce --quick
```

Run ablation variants for the comparison tables:

```powershell
python run_experiment.py --data-path path\to\pv_fault_data.csv --scenario scarce --variant baseline
python run_experiment.py --data-path path\to\pv_fault_data.csv --scenario scarce --variant no_feedback
python run_experiment.py --data-path path\to\pv_fault_data.csv --scenario scarce --variant no_difficulty
python run_experiment.py --data-path path\to\pv_fault_data.csv --scenario scarce --variant no_dual_gate
python run_experiment.py --data-path path\to\pv_fault_data.csv --scenario scarce --variant no_convergence
```

Run generator-pool experiments by selecting a subset:

```powershell
python run_experiment.py --data-path path\to\pv_fault_data.csv --scenario scarce --generators wgan_gp,ctgan,cvae
```

## Additional experiments reported in the manuscript

### Baseline comparison and cost table

`run_baselines.py` evaluates the proposed framework next to the baselines used
in the comparison table: no augmentation, SMOTE, one-shot WGAN-GP/CVAE/DDPM,
the static equal-weight ensemble, GAN+SMOTE, uncertainty-guided adaptive
augmentation, and curriculum-style adaptive augmentation.  It writes per-run
metrics, a comparison table, wall-clock cost, and run-matched paired
differences against the proposed framework.

```powershell
python run_baselines.py --data-path path\to\pv_fault_data.csv --scenario all --runs 5
```

Outputs: `runs.csv`, `comparison_table.csv`, `summary.json`,
`paired_vs_proposed.csv`, and `predictions/` (per-run test predictions).

### Generator-pool comparison under a fixed budget

`run_generator_pool.py` compares individual generators and cumulative
combinations.  `--mode fixed` allocates an identical accepted-sample budget to
every pool (the controlled comparison); `--mode natural` runs the full
closed-loop policy with each pool, so larger pools accept more samples.

```powershell
python run_generator_pool.py --data-path path\to\pv_fault_data.csv --scenario scarce --mode fixed --budget 500
python run_generator_pool.py --data-path path\to\pv_fault_data.csv --scenario scarce --mode natural
```

### Error analysis, robustness, and physical checks

`run_analysis.py` produces the confusion matrices, class-pair table, operating
regime analysis, measurement/domain perturbations, group-held-out shifts, and
real-versus-generated physical parameters.

```powershell
python run_analysis.py --data-path path\to\pv_fault_data.csv --scenario scarce --runs 5
```

Optional metadata columns enable the regime and group-held-out experiments.  A
CSV may include `irradiance`, `temperature`, `module_id`, and `condition_id`;
an NPZ may store them as arrays with the same names.  Pass
`--metadata-columns irradiance,temperature,module_id,condition_id` (the default)
so they are excluded from the features and carried through the split.  Use
`--voltage-max` and `--current-max` to express physical parameters in volts,
amperes, and watts; otherwise the parameters are reported in normalised units.

Outputs: `analysis_summary.json` and `figures/` (`confusion_matrices.png`,
`real_vs_generated.png`).

### Alternative classifiers

All scripts accept `--classifier {cnn1d,resnet1d,bilstm}` for the
classifier-generality experiment.

```powershell
python run_baselines.py --data-path path\to\pv_fault_data.csv --scenario scarce --classifier resnet1d --runs 5
python run_experiment.py --data-path path\to\pv_fault_data.csv --scenario scarce --classifier bilstm
```

### Run-level figures

`make_figures.py` builds the convergence and allocation figures from the
`{scenario}_run_*` directories written by `run_experiment.py`.

```powershell
python make_figures.py --runs-root outputs --output-dir outputs\figures
```

## Implemented modules

- `pv_fault_aug/data.py`: CSV/NPZ loading, balanced/imbalanced/scarce scenario construction, stratified train/validation/test split.
- `pv_fault_aug/models/classifier.py`: 1D-CNN downstream classifier.
- `pv_fault_aug/models/generators.py`: WGAN-GP, conditional GAN as CTGAN-style generator, CVAE, VAE-GAN, DDPM, and conditional DDPM.
- `pv_fault_aug/feedback.py`: per-class error, entropy confidence, and scarcity feedback.
- `pv_fault_aug/allocation.py`: difficulty-adaptive class quota allocation and generator-class quota splitting.
- `pv_fault_aug/mopso.py`: compact MOPSO implementation for generator global weights.
- `pv_fault_aug/screening.py`: MMD/Wasserstein distributional gate and F1-gain performance gate.
- `pv_fault_aug/pipeline.py`: closed-loop iterative augmentation process and result export.
- `pv_fault_aug/baselines.py`: SMOTE, one-shot generation, static ensemble, GAN+SMOTE, uncertainty-guided and curriculum-style baselines under the same splits.
- `pv_fault_aug/analysis.py`: confusion/class-pair analysis, operating regimes, measurement and domain perturbations, group-held-out splits, and physical I-V parameters.
- `pv_fault_aug/stats.py`: run-matched paired differences and bootstrap confidence intervals (descriptive, no p-values).
- `pv_fault_aug/figures.py`: run-level convergence/allocation, confusion-matrix, and real-versus-generated figures.

## Paper correspondence

The implementation follows Algorithm 1 in the manuscript:

- The real training split initializes `D_aug`, while validation and test data are never augmented.
- I-V features are min-max normalized with training-set statistics after the stratified split.
- Per-class error, entropy confidence, and scarcity form the feedback vector used for difficulty weights.
- MOPSO selects global generator weights from probing quotas using distributional quality, validation F1, and entropy objectives.
- Class quotas and generator-class quotas are updated before candidate generation.
- Candidate samples pass class-wise MMD/Wasserstein screening and the class-bound marginal F1 gate before entering `D_aug`.
- Convergence uses round-level F1 saturation together with feature-center drift, matching the joint stopping rule in the paper.

`--variant baseline` trains the classifier directly on the selected scenario
without synthetic augmentation. The other variants keep the same data split and
disable one mechanism at a time, matching the ablation logic in the paper.

## Notes

The repository stores the runnable configuration used by the manuscript,
including split ratios, random seeds, generator settings, screening thresholds,
and convergence-control parameters. Each experiment writes the resolved
configuration to `config.json` in the output directory.

The full DDPM and six-generator loop is computationally expensive. Use `--quick`
first after the dataset is available, then run the full setting for final
tables and figures.
