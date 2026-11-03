import logging
import os

import hydra
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf

import wandb
from simple_metalearning.datasets.dataset import MetaDatasetFactory
from simple_metalearning.datasets.metafeature_transforms import (
    apply_metafeature_transforms_from_config,
)
from simple_metalearning.experiments.base import Experiment

log = logging.getLogger(__name__)

# Quiet the noisiest third-party loggers. AutoGluon prints a full training
# report per fit (≈50 lines) and Ray is chatty; across thousands of live
# evaluations that bloats logs (and wandb's console capture) enormously.
for _noisy in ("autogluon", "ray", "lightgbm", "xgboost"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    # Don't let wandb mirror all stdout/stderr into the run dir — the console
    # capture is the main reason offline run dirs balloon. The SLURM .out (or
    # terminal) already has the console; wandb only needs metrics/artifacts.
    os.environ.setdefault("WANDB_CONSOLE", "off")

    # Setup WandB
    wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        settings=wandb.Settings(console="off"),
        **cfg.wandb,
    )

    # Resolve absolute path for cache
    if hasattr(cfg.dataset, "cache_dir"):
        cfg.dataset.cache_dir = to_absolute_path(cfg.dataset.cache_dir)

    # Load Dataset
    log.info(f"Loading dataset: {cfg.dataset._target_}")
    dataset_factory: MetaDatasetFactory = instantiate(cfg.dataset)
    full_dataset = dataset_factory.load()
    log.info(f"Dataset loaded: {full_dataset.dataset_name}")
    log.info(f"Shape: {full_dataset.performance.shape} (Flows x Tasks)")
    wandb.log(
        {
            "dataset_name": full_dataset.dataset_name,
            "n_flows": full_dataset.performance.shape[0],
            "n_tasks": full_dataset.performance.shape[1],
        }
    )

    # Apply metafeature transformations if configured
    if cfg.get("metafeature_transforms", None) is not None:
        log.info("Applying metafeature transformations...")
        full_dataset = apply_metafeature_transforms_from_config(
            full_dataset, cfg.metafeature_transforms
        )
        log.info(f"Transformed flow metadata shape: {full_dataset.flow_metadata.shape}")
        log.info(f"Transformed task metadata shape: {full_dataset.task_metadata.shape}")

    # Instantiate and run experiment
    log.info(f"Running experiment: {cfg.experiment._target_}")

    # Instantiate estimator factory (partial function)
    estimator_factory = instantiate(cfg.estimator)

    # Instantiate experiment
    experiment: Experiment = instantiate(cfg.experiment)
    results = experiment.run(dataset=full_dataset, estimator_factory=estimator_factory)

    log.info("Experiment completed successfully.")
    wandb.finish()

    return results


if __name__ == "__main__":
    main()
