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

From `论文代码`:

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

## Implemented modules

- `pv_fault_aug/data.py`: CSV/NPZ loading, balanced/imbalanced/scarce scenario construction, stratified train/validation/test split.
- `pv_fault_aug/models/classifier.py`: 1D-CNN downstream classifier.
- `pv_fault_aug/models/generators.py`: WGAN-GP, conditional GAN as CTGAN-style generator, CVAE, VAE-GAN, DDPM, and conditional DDPM.
- `pv_fault_aug/feedback.py`: per-class error, entropy confidence, and scarcity feedback.
- `pv_fault_aug/allocation.py`: difficulty-adaptive class quota allocation and generator-class quota splitting.
- `pv_fault_aug/mopso.py`: compact MOPSO implementation for generator global weights.
- `pv_fault_aug/screening.py`: MMD/Wasserstein distributional gate and F1-gain performance gate.
- `pv_fault_aug/pipeline.py`: closed-loop iterative augmentation process and result export.

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
