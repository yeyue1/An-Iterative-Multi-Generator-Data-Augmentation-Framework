from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


FAULT_CLASSES = ["NS", "SC", "OC", "PS", "DOA", "DOA+SC", "DOA+OC"]
SUPPORTED_GENERATORS = ("wgan_gp", "ctgan", "cvae", "vaegan", "ddpm", "cddpm")
SUPPORTED_CLASSIFIERS = ("cnn1d", "resnet1d", "bilstm")
SUPPORTED_VARIANTS = (
    "full",
    "baseline",
    "no_feedback",
    "no_difficulty",
    "no_dual_gate",
    "no_convergence",
)


@dataclass
class DataConfig:
    data_path: Path | None = None
    label_column: str = "label"
    feature_prefix: str | None = None
    feature_dim: int = 100
    train_ratio: float = 0.60
    val_ratio: float = 0.20
    test_ratio: float = 0.20
    scarce_ratio: float = 0.10
    random_seed: int = 42
    n_runs: int = 5
    class_names: list[str] = field(default_factory=lambda: FAULT_CLASSES.copy())
    metadata_columns: tuple[str, ...] = ()
    irradiance_column: str = "irradiance"
    temperature_column: str = "temperature"
    module_column: str = "module_id"
    condition_column: str = "condition_id"


@dataclass
class ClassifierConfig:
    architecture: str = "cnn1d"
    epochs: int = 200
    feedback_epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 20
    dropout: float = 0.5


@dataclass
class GeneratorConfig:
    names: tuple[str, ...] = SUPPORTED_GENERATORS
    latent_dim: int = 32
    hidden_dims: tuple[int, ...] = (256, 512, 256)
    epochs: int = 500
    quick_epochs: int = 80
    batch_size: int = 128
    learning_rate: float = 2e-4
    beta_vae: float = 4.0
    gradient_penalty: float = 10.0
    critic_steps: int = 5
    diffusion_steps: int = 1000
    diffusion_sample_steps: int = 50
    guidance_scale: float = 3.0


@dataclass
class FrameworkConfig:
    alpha_error: float = 0.4
    alpha_uncertainty: float = 0.3
    alpha_scarcity: float = 0.3
    mmd_threshold: float = 0.10
    wasserstein_threshold: float = 0.05
    min_f1_gain: float = 0.005
    gate_utility_weight: float = 0.7
    macro_f1_degradation_tolerance: float = 0.002
    gate_eval_runs: int = 3
    max_iterations: int = 30
    f1_saturation_window: int = 5
    center_drift_threshold: float = 0.01
    max_per_round_generation_ratio: float = 5.0
    min_samples_per_class: int = 50
    base_quota: int = 32
    min_quota: int = 4
    max_quota: int = 256
    use_feedback_allocation: bool = True
    use_difficulty_allocation: bool = True
    use_dual_gate: bool = True
    use_convergence_control: bool = True


@dataclass
class MopsoConfig:
    swarm_size: int = 50
    iterations: int = 80
    inertia: float = 0.7
    cognitive: float = 1.5
    social: float = 1.5


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    generators: GeneratorConfig = field(default_factory=GeneratorConfig)
    framework: FrameworkConfig = field(default_factory=FrameworkConfig)
    mopso: MopsoConfig = field(default_factory=MopsoConfig)
    output_dir: Path = Path("outputs")
    device: str = "auto"
    scenario: str = "scarce"
    variant: str = "full"

    def apply_variant(self) -> None:
        if self.variant == "baseline":
            self.framework.max_iterations = 0
        elif self.variant == "no_feedback":
            self.framework.use_feedback_allocation = False
        elif self.variant == "no_difficulty":
            self.framework.use_difficulty_allocation = False
        elif self.variant == "no_dual_gate":
            self.framework.use_dual_gate = False
        elif self.variant == "no_convergence":
            self.framework.use_convergence_control = False

    def validate(self) -> None:
        split_total = self.data.train_ratio + self.data.val_ratio + self.data.test_ratio
        if abs(split_total - 1.0) > 1e-6:
            raise ValueError(f"Train/val/test ratios must sum to 1.0, got {split_total}")
        alpha_total = (
            self.framework.alpha_error
            + self.framework.alpha_uncertainty
            + self.framework.alpha_scarcity
        )
        if abs(alpha_total - 1.0) > 1e-6:
            raise ValueError(f"Difficulty coefficients must sum to 1.0, got {alpha_total}")
        if self.scenario not in {"balanced", "imbalanced", "scarce"}:
            raise ValueError("scenario must be one of: balanced, imbalanced, scarce")
        if self.variant not in SUPPORTED_VARIANTS:
            raise ValueError(f"variant must be one of: {', '.join(SUPPORTED_VARIANTS)}")
        if self.classifier.architecture not in SUPPORTED_CLASSIFIERS:
            raise ValueError(
                f"classifier.architecture must be one of: {', '.join(SUPPORTED_CLASSIFIERS)}"
            )
        unknown_generators = sorted(set(self.generators.names) - set(SUPPORTED_GENERATORS))
        if unknown_generators:
            raise ValueError(f"Unknown generators: {unknown_generators}")
        if len(self.generators.names) == 0 and self.variant != "baseline":
            raise ValueError("At least one generator is required unless variant is baseline")
        if not 0.0 <= self.framework.gate_utility_weight <= 1.0:
            raise ValueError("gate_utility_weight must be in [0, 1]")
        if self.framework.gate_eval_runs < 1:
            raise ValueError("gate_eval_runs must be at least 1")
