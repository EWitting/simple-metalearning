"""Task-only BoTorch Gaussian Process estimator with Expected Improvement.

Drop-in replacement for TaskOnlyGPEIEstimator using BoTorch's SingleTaskGP
(GPyTorch backend) instead of sklearn's GaussianProcessRegressor.

Key improvements over the sklearn GP:
- Matern-5/2 kernel (better for non-smooth HPO landscapes)
- ARD lengthscales by default (learns which hyperparameters matter)
- Gamma priors on lengthscale and noise (prevents degenerate fits with few obs)
- Normalize + Standardize input/output transforms (numerical stability)
- Warm-starts from previous optimised hyperparameters (faster incremental fits)
"""

import warnings
from typing import Optional, Union, List

import numpy as np
import torch

from ..datasets.dataset import MetaDataset
from ..datasets.metafeature_transforms import PerformanceVectorTransform
from .estimator import Estimator
from .acquisition import AcquisitionFunction, EIGaussianAcquisition


class TaskOnlyBoTorchGPEIEstimator(Estimator):
    """BoTorch GP estimator that trains per-task models using flow metafeatures.

    Args:
        dataset: MetaDataset with performance matrix and flow metadata.
        n_restarts: Number of random restarts for MLL optimisation (0 = warm-start only).
        f_star_from_predictions: If False (default), f_star is the best observed
            value. Set to True to use the best GP-predicted mean.
        acquisition_fn: Acquisition function. Defaults to EIGaussianAcquisition.
        performance_transform: Optional PerformanceVectorTransform for collaborative
            filtering features (fitted on masked matrix inside fit()).
        use_metadata: Include flow_metadata in GP inputs. Defaults to True.
        normalize_inputs: Apply BoTorch Normalize input transform. Defaults to True.
        standardize_outputs: Apply BoTorch Standardize output transform. Defaults to True.
    """

    def __init__(
        self,
        dataset: MetaDataset,
        n_restarts: int = 0,
        f_star_from_predictions: bool = False,
        acquisition_fn: Optional[AcquisitionFunction] = None,
        performance_transform: Optional[PerformanceVectorTransform] = None,
        use_metadata: bool = True,
        normalize_inputs: bool = True,
        standardize_outputs: bool = True,
    ):
        super().__init__(dataset, acquisition_fn or EIGaussianAcquisition())
        self.n_restarts = n_restarts
        self.f_star_from_predictions = f_star_from_predictions
        self.performance_transform = performance_transform
        self.use_metadata = use_metadata
        self.normalize_inputs = normalize_inputs
        self.standardize_outputs = standardize_outputs

        self.n_flows = len(dataset.performance)
        self.n_tasks = len(dataset.performance.columns)

        if (not use_metadata or dataset.flow_metadata is None) and performance_transform is None:
            raise ValueError(
                "TaskOnlyBoTorchGPEIEstimator requires either flow_metadata "
                "(with use_metadata=True) or a performance_transform (or both)."
            )

        self.task_models_: dict[int, object] = {}
        # Stores optimised kernel hyperparameters for warm-starting per task
        self.task_hyperparams_: dict[int, dict] = {}
        # Cache (mu, sigma) per task; invalidated by partial_fit
        self._prediction_cache_: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        self.fit()

    def fit(self) -> None:
        """Fit the performance transform (if any); per-task GPs are fitted lazily."""
        if self.performance_transform is not None:
            self.performance_transform.fit(self.dataset.performance)
        self.task_models_.clear()
        self.task_hyperparams_.clear()
        self._prediction_cache_.clear()

    def _get_flow_features(self) -> np.ndarray:
        parts = []
        if self.use_metadata and self.dataset.flow_metadata is not None:
            parts.append(self.dataset.flow_metadata.values.astype(float))
        if self.performance_transform is not None:
            parts.append(self.performance_transform.transform(self.dataset.performance))
        return np.concatenate(parts, axis=1)

    def _fit_task(self, task_idx: int) -> None:
        """Fit a SingleTaskGP for the given task on its current observations."""
        from botorch.models import SingleTaskGP
        from botorch.models.transforms.input import Normalize
        from botorch.models.transforms.outcome import Standardize
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood

        task_performance = self.dataset.performance.iloc[:, task_idx].values
        observed_mask = ~np.isnan(task_performance)
        n_obs = observed_mask.sum()

        if n_obs < 2:
            return

        X_all = self._get_flow_features()
        y_obs = task_performance[observed_mask]

        if np.std(y_obs) < 1e-10:
            return

        # Sanitize: BoTorch raises InputDataError on any NaN/inf in train_X
        X_obs = np.nan_to_num(X_all[observed_mask], nan=0.0, posinf=0.0, neginf=0.0)
        X_tensor = torch.tensor(X_obs, dtype=torch.float64)
        y_tensor = torch.tensor(y_obs, dtype=torch.float64).unsqueeze(-1)

        d = X_tensor.shape[1]

        model = None
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                input_transform = Normalize(d=d) if self.normalize_inputs else None
                outcome_transform = Standardize(m=1) if self.standardize_outputs else None

                model = SingleTaskGP(
                    train_X=X_tensor,
                    train_Y=y_tensor,
                    input_transform=input_transform,
                    outcome_transform=outcome_transform,
                )

                # Warm-start: initialise from previous optimised hyperparameters
                if task_idx in self.task_hyperparams_:
                    prev = self.task_hyperparams_[task_idx]
                    try:
                        model.covar_module.base_kernel.lengthscale = prev["lengthscale"]
                        model.covar_module.outputscale = prev["outputscale"]
                        model.likelihood.noise = prev["noise"]
                    except Exception:
                        pass

                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_mll(mll)

                # Optional random restarts to escape local optima in MLL landscape
                if self.n_restarts > 0:
                    model.train()
                    with torch.no_grad():
                        best_loss = -mll(model.likelihood(model(X_tensor)), y_tensor).item()
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}

                    for _ in range(self.n_restarts):
                        try:
                            model_r = SingleTaskGP(
                                train_X=X_tensor,
                                train_Y=y_tensor,
                                input_transform=Normalize(d=d) if self.normalize_inputs else None,
                                outcome_transform=Standardize(m=1) if self.standardize_outputs else None,
                            )
                            mll_r = ExactMarginalLogLikelihood(model_r.likelihood, model_r)
                            fit_gpytorch_mll(mll_r)
                            model_r.train()
                            with torch.no_grad():
                                loss_r = -mll_r(model_r.likelihood(model_r(X_tensor)), y_tensor).item()
                            if loss_r < best_loss:
                                best_loss = loss_r
                                best_state = {k: v.clone() for k, v in model_r.state_dict().items()}
                        except Exception:
                            pass

                    model.load_state_dict(best_state)

            except Exception:
                model = None

        if model is None:
            return

        model.eval()
        self.task_models_[task_idx] = model
        self._prediction_cache_.pop(task_idx, None)

        try:
            self.task_hyperparams_[task_idx] = {
                "lengthscale": model.covar_module.base_kernel.lengthscale.detach().clone(),
                "outputscale": model.covar_module.outputscale.detach().clone(),
                "noise": model.likelihood.noise.detach().clone(),
            }
        except Exception:
            pass

    def _predict_with_model(self, task_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (mu, sigma) for all flows; uses a per-task cache."""
        if task_idx in self._prediction_cache_:
            return self._prediction_cache_[task_idx]

        model = self.task_models_[task_idx]
        X_tensor = torch.tensor(
            np.nan_to_num(self._get_flow_features(), nan=0.0, posinf=0.0, neginf=0.0),
            dtype=torch.float64,
        )

        model.eval()
        with torch.no_grad():
            posterior = model.posterior(X_tensor)
            mu = posterior.mean.squeeze(-1).numpy()
            sigma = posterior.variance.clamp(min=0.0).sqrt().squeeze(-1).numpy()

        self._prediction_cache_[task_idx] = (mu, sigma)
        return mu, sigma

    def predict(self, task_index: int) -> np.ndarray:
        if task_index not in self.task_models_:
            self._fit_task(task_index)
        if task_index not in self.task_models_:
            return np.zeros(self.n_flows)
        mu, _ = self._predict_with_model(task_index)
        return mu

    def predict_uncertainty(self, task_index: int) -> Optional[np.ndarray]:
        if task_index not in self.task_models_:
            self._fit_task(task_index)
        if task_index not in self.task_models_:
            return None
        _, sigma = self._predict_with_model(task_index)
        return sigma

    def suggest(
        self, task_index: int, predictions: Optional[Union[np.ndarray, List[float]]] = None
    ) -> np.ndarray:
        """Suggest the next flow using the configured acquisition function."""
        if task_index not in self.task_models_:
            self._fit_task(task_index)

        if not self.f_star_from_predictions:
            return super().suggest(task_index, predictions)

        # f_star_from_predictions=True: use best GP-predicted mean over known flows
        perf_col = self.dataset.performance.iloc[:, task_index].values
        known_mask = ~np.isnan(perf_col)

        if task_index in self.task_models_:
            mu, sigma = self._predict_with_model(task_index)
        else:
            mu = np.zeros(self.n_flows)
            sigma = None

        if predictions is not None:
            mu = np.asarray(predictions)

        if np.any(known_mask) and task_index in self.task_models_:
            X_known = torch.tensor(self._get_flow_features()[known_mask], dtype=torch.float64)
            model = self.task_models_[task_index]
            model.eval()
            with torch.no_grad():
                f_star = float(model.posterior(X_known).mean.max().item())
        elif np.any(known_mask):
            f_star = float(np.max(perf_col[known_mask]))
        else:
            f_star = float("-inf")

        return self.acquisition_fn.compute(mu, sigma, f_star, known_mask)

    def partial_fit(
        self,
        flow_index: Union[int, List[int]],
        task_index: int,
        performance: Union[float, List[float]],
    ) -> None:
        self.dataset.performance.iloc[flow_index, task_index] = performance
        self._prediction_cache_.pop(task_index, None)
        self._fit_task(task_index)
