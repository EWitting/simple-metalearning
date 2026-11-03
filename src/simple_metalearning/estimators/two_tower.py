"""Two-tower collaborative/content-based estimator with PyTorch.

Learns separate representations (towers) for flows and tasks, and predicts
performance via dot-product similarity between the resulting embeddings.

Each tower can be:
  - A linear map (single layer) — backward-compatible default
  - A configurable MLP (multiple hidden layers + activation)

Each side optionally adds a learnable ``nn.Embedding`` (pure collaborative
filtering) to the content features.  Content and embedding are concatenated
before being passed through the tower.

+-------------------+        +-----------------+
|  flow content     |        |  task content   |
|  (optional)       |   ...  |  (optional)     |
|  flow embedding   |        |  task embedding |
|  (optional)       |        |  (optional)     |
+-------------------+        +-----------------+
         |                            |
  flow MLP tower              task MLP tower
         |                            |
   flow_vec (d)              task_vec (d)
         |___________________________|
                   dot product
                   → scalar score
"""

from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn

from ..datasets.dataset import MetaDataset
from .acquisition import AcquisitionFunction, EmbeddingMMRAcquisition
from .estimator import Estimator
from .losses.info_nce import InfoNCELossWithEmbeddings
from .torch_trainer import TorchTrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_mlp(
    in_dim: int,
    hidden_dims: Optional[Sequence[int]],
    out_dim: int,
    activation: nn.Module,
) -> nn.Sequential:
    """Build a linear or MLP tower.

    * ``hidden_dims=None``  — no tower; input passes through unchanged
      (``nn.Identity``).  Requires ``in_dim == out_dim``.
    * ``hidden_dims=[]``    — single ``nn.Linear`` (no activation).
    * ``hidden_dims=[h…]``  — MLP with *activation* between layers.
    """
    if hidden_dims is None:
        if in_dim != out_dim:
            raise ValueError(
                f"hidden_dims=None (identity tower) requires in_dim == out_dim, "
                f"got {in_dim} != {out_dim}."
            )
        return nn.Sequential(nn.Identity())

    if not hidden_dims:
        return nn.Sequential(nn.Linear(in_dim, out_dim))

    layers: list[nn.Module] = []
    current = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(current, h))
        layers.append(activation)
        current = h
    layers.append(nn.Linear(current, out_dim))
    return nn.Sequential(*layers)


_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "leaky_relu": nn.LeakyReLU,
    "silu": nn.SiLU,
}


# ---------------------------------------------------------------------------
# Neural network module
# ---------------------------------------------------------------------------


class TwoTowerModel(nn.Module):
    """Two-tower neural network for flow–task performance prediction.

    Each "side" (flow / task) builds its input vector by concatenating
    whichever sources are enabled, then passes the result through an MLP:

    * **Content features**: raw metadata columns (``n_*_features`` > 0).
    * **Learned embedding**: ``nn.Embedding`` lookup (``*_embed_dim`` > 0).

    At least one source must be enabled per side.

    Args:
        n_flow_features: Number of flow content-feature columns (0 = none).
        n_task_features: Number of task content-feature columns (0 = none).
        embedding_dim: Output dimensionality of both towers (shared).
        n_flows: Total number of flows; required when ``flow_embed_dim > 0``.
        n_tasks: Total number of tasks; required when ``task_embed_dim > 0``.
        flow_embed_dim: Dimension of the learnable flow embedding (0 = off).
        task_embed_dim: Dimension of the learnable task embedding (0 = off).
        flow_tower_dims: Hidden layer widths for the flow MLP.  Empty tuple
            → single linear (default, backward-compatible).
        task_tower_dims: Hidden layer widths for the task MLP.
        activation: Name of the nonlinearity between hidden layers.
            One of ``"relu"`` (default), ``"gelu"``, ``"tanh"``,
            ``"leaky_relu"``, ``"silu"``.
    """

    def __init__(
        self,
        n_flow_features: int,
        n_task_features: int,
        embedding_dim: int,
        n_flows: int = 0,
        n_tasks: int = 0,
        flow_embed_dim: int = 0,
        task_embed_dim: int = 0,
        flow_tower_dims: Optional[Sequence[int]] = (),
        task_tower_dims: Optional[Sequence[int]] = (),
        activation: str = "relu",
        use_flow_features: bool = True,
        use_task_features: bool = True,
    ) -> None:
        super().__init__()

        # Suppress content features when running in embed-only mode.
        n_flow_features = n_flow_features if use_flow_features else 0
        n_task_features = n_task_features if use_task_features else 0

        self.n_flow_features = n_flow_features
        self.n_task_features = n_task_features
        self.flow_embed_dim = flow_embed_dim
        self.task_embed_dim = task_embed_dim
        self.embedding_dim = embedding_dim

        # Learned embedding tables (optional)
        self.flow_embedding: Optional[nn.Embedding] = None
        self.task_embedding: Optional[nn.Embedding] = None
        if flow_embed_dim > 0:
            if n_flows <= 0:
                raise ValueError("n_flows must be > 0 when flow_embed_dim > 0.")
            self.flow_embedding = nn.Embedding(n_flows, flow_embed_dim)
        if task_embed_dim > 0:
            if n_tasks <= 0:
                raise ValueError("n_tasks must be > 0 when task_embed_dim > 0.")
            self.task_embedding = nn.Embedding(n_tasks, task_embed_dim)

        flow_in_dim = n_flow_features + flow_embed_dim
        task_in_dim = n_task_features + task_embed_dim

        if flow_in_dim <= 0:
            raise ValueError(
                "Flow tower has no input: set n_flow_features > 0 or flow_embed_dim > 0."
            )
        if task_in_dim <= 0:
            raise ValueError(
                "Task tower has no input: set n_task_features > 0 or task_embed_dim > 0."
            )

        act_cls = _ACTIVATIONS.get(activation.lower())
        if act_cls is None:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Choose from: {list(_ACTIVATIONS)}."
            )
        act = act_cls()

        self.flow_tower = _build_mlp(flow_in_dim, flow_tower_dims, embedding_dim, act)
        self.task_tower = _build_mlp(task_in_dim, task_tower_dims, embedding_dim, act)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flow_input(
        self,
        features: Optional[torch.Tensor],
        indices: Optional[torch.Tensor],
    ) -> torch.Tensor:
        parts = []
        if features is not None and self.n_flow_features > 0:
            parts.append(features)
        if self.flow_embedding is not None:
            assert indices is not None, "flow_indices required when flow_embed_dim > 0"
            parts.append(self.flow_embedding(indices))
        if not parts:
            raise RuntimeError("Flow input is empty — no features and no embedding.")
        return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]

    def _task_input(
        self,
        features: Optional[torch.Tensor],
        indices: Optional[torch.Tensor],
    ) -> torch.Tensor:
        parts = []
        if features is not None and self.n_task_features > 0:
            parts.append(features)
        if self.task_embedding is not None:
            assert indices is not None, "task_indices required when task_embed_dim > 0"
            parts.append(self.task_embedding(indices))
        if not parts:
            raise RuntimeError("Task input is empty — no features and no embedding.")
        return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        flow_features: Optional[torch.Tensor],
        task_features: Optional[torch.Tensor],
        flow_indices: Optional[torch.Tensor] = None,
        task_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict performance scores for a batch of (flow, task) pairs.

        Args:
            flow_features: ``(batch, n_flow_features)`` or ``None``.
            task_features: ``(batch, n_task_features)`` or ``None``.
            flow_indices: ``(batch,)`` integer tensor; required when
                ``flow_embed_dim > 0``.
            task_indices: ``(batch,)`` integer tensor; required when
                ``task_embed_dim > 0``.

        Returns:
            ``(batch,)`` predicted scores.
        """
        flow_vec = self.flow_tower(self._flow_input(flow_features, flow_indices))
        task_vec = self.task_tower(self._task_input(task_features, task_indices))
        return (flow_vec * task_vec).sum(dim=1)

    def get_flow_embeddings(
        self,
        flow_features: Optional[torch.Tensor],
        flow_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return tower output for all flows.  Shape: ``(n_flows, embedding_dim)``."""
        return self.flow_tower(self._flow_input(flow_features, flow_indices))

    def get_task_embedding(
        self,
        task_features: Optional[torch.Tensor],
        task_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return tower output for one or more tasks.  Shape: ``(n, embedding_dim)``."""
        return self.task_tower(self._task_input(task_features, task_indices))


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


class TwoTowerEstimator(Estimator):
    """Two-tower estimator that learns flow/task representations via backprop.

    Supports three representation modes (configurable per side):

    * **Content-only**: flow/task metadata → MLP → embedding.
    * **Embedding-only**: learned ``nn.Embedding`` → MLP → embedding
      (no content features required).
    * **Content + embedding**: concatenate metadata and learned embedding
      before the MLP.

    Args:
        dataset: MetaDataset.  ``flow_metadata`` / ``task_metadata`` are
            required unless the corresponding ``*_embed_dim > 0`` is set
            and pure-embedding mode is intended.
        embedding_dim: Output dimension of both towers.
        flow_embed_dim: Dimension of the learnable per-flow embedding.
            0 (default) = no learned embedding.
        task_embed_dim: Dimension of the learnable per-task embedding.
            0 (default) = no learned embedding.
        flow_tower_dims: Hidden layer widths for the flow MLP.  Empty →
            single linear (backward-compatible default).
        task_tower_dims: Hidden layer widths for the task MLP.
        activation: Nonlinearity name (``"relu"``, ``"gelu"``, etc.).
        n_epochs: Training epochs for the initial ``fit()`` call.
        n_iter_update: Fine-tuning epochs in ``partial_fit()``.  The model
            weights are kept (warm restart); only a fresh optimizer is created.
            Defaults to 10 % of *n_epochs*.
        batch_size: Mini-batch size.
        loss_fn: PyTorch loss module.  Defaults to ``nn.MSELoss()``.
        optimizer_class: Callable that returns an ``Optimizer`` given
            ``model.parameters()``.  Defaults to ``torch.optim.AdamW``.
        scheduler_class: Optional LR scheduler factory.
        acquisition_fn: Acquisition function for the BO loop.  Defaults to
            :class:`~simple_metalearning.estimators.acquisition.GreedyAcquisition`.
        device: ``"auto"``, ``"cuda"``, or ``"cpu"``.
        random_state: Random seed.
        verbose: Show tqdm progress bar.
    """

    def __init__(
        self,
        dataset: MetaDataset,
        embedding_dim: int = 32,
        flow_embed_dim: int = 0,
        task_embed_dim: int = 0,
        flow_tower_dims: Optional[Sequence[int]] = (),
        task_tower_dims: Optional[Sequence[int]] = (),
        activation: str = "relu",
        use_flow_features: bool = True,
        use_task_features: bool = True,
        n_epochs: int = 100,
        n_iter_update: int = 10,
        update_task_only: bool = False,
        batch_size: int = 512,
        loss_fn: Optional[nn.Module] = None,
        optimizer_class=None,
        scheduler_class=None,
        acquisition_fn: Optional[AcquisitionFunction] = None,
        device: str = "auto",
        random_state: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(dataset, acquisition_fn)

        perf = dataset.require_matrix()
        self.n_flows = len(perf)
        self.n_tasks = len(perf.columns)

        n_flow_features = dataset.flow_metadata.shape[1] if (use_flow_features and dataset.flow_metadata is not None) else 0
        n_task_features = dataset.task_metadata.shape[1] if (use_task_features and dataset.task_metadata is not None) else 0

        if n_flow_features == 0 and flow_embed_dim == 0:
            raise ValueError(
                "TwoTowerEstimator: flow side needs either flow_metadata or flow_embed_dim > 0."
            )
        if n_task_features == 0 and task_embed_dim == 0:
            raise ValueError(
                "TwoTowerEstimator: task side needs either task_metadata or task_embed_dim > 0."
            )

        self.use_flow_features = use_flow_features
        self.use_task_features = use_task_features
        self.n_flow_features = n_flow_features
        self.n_task_features = n_task_features
        self.embedding_dim = embedding_dim
        self.flow_embed_dim = flow_embed_dim
        self.task_embed_dim = task_embed_dim
        self.flow_tower_dims = None if flow_tower_dims is None else tuple(flow_tower_dims)
        self.task_tower_dims = None if task_tower_dims is None else tuple(task_tower_dims)
        self.activation = activation
        self.n_epochs = n_epochs
        self.n_iter_update = n_iter_update
        self.update_task_only = update_task_only
        self.batch_size = batch_size
        self.loss_fn: nn.Module = loss_fn if loss_fn is not None else nn.MSELoss()
        self.optimizer_class = optimizer_class if optimizer_class is not None else torch.optim.AdamW
        self.scheduler_class = scheduler_class
        self.random_state = random_state
        self.verbose = verbose

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device in ("cuda", "cpu"):
            self.device = torch.device(device)
        else:
            raise ValueError(f"Invalid device '{device}'. Use 'auto', 'cuda', or 'cpu'.")

        if self.verbose:
            print(f"TwoTowerEstimator on {self.device}")

        self.model: Optional[TwoTowerModel] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler = None
        self._flow_feat_tensor: Optional[torch.Tensor] = None
        self._task_feat_tensor: Optional[torch.Tensor] = None
        self._all_flow_indices: Optional[torch.Tensor] = None
        self._all_task_indices: Optional[torch.Tensor] = None

        self.fit()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _to_tensor(self, arr: Optional[np.ndarray], dtype=torch.float32) -> Optional[torch.Tensor]:
        if arr is None:
            return None
        return torch.tensor(arr.astype(np.float32), dtype=dtype, device=self.device)

    def _cache_tensors(self) -> None:
        """Pre-compute and cache feature tensors (and index ranges)."""
        ds = self.dataset
        self._flow_feat_tensor = (
            self._to_tensor(ds.flow_metadata.values)
            if (self.use_flow_features and ds.flow_metadata is not None)
            else None
        )
        self._task_feat_tensor = (
            self._to_tensor(ds.task_metadata.values)
            if (self.use_task_features and ds.task_metadata is not None)
            else None
        )
        self._all_flow_indices = torch.arange(self.n_flows, device=self.device)
        self._all_task_indices = torch.arange(self.n_tasks, device=self.device)

    def _prepare_training_data(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Collect observed (flow_idx, task_idx, performance) triplets."""
        perf = self.dataset.require_matrix().values
        fi_list, ti_list, y_list = [], [], []
        for fi in range(self.n_flows):
            for ti in range(self.n_tasks):
                v = perf[fi, ti]
                if not np.isnan(v):
                    fi_list.append(fi)
                    ti_list.append(ti)
                    y_list.append(v)
        flow_idx = torch.tensor(fi_list, dtype=torch.long, device=self.device)
        task_idx = torch.tensor(ti_list, dtype=torch.long, device=self.device)
        targets = torch.tensor(y_list, dtype=torch.float32, device=self.device)
        return flow_idx, task_idx, targets

    # ------------------------------------------------------------------
    # Estimator interface
    # ------------------------------------------------------------------

    def fit(self) -> None:
        """Reinitialise the model from scratch and train for ``n_epochs``."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)

        self._cache_tensors()

        self.model = TwoTowerModel(
            n_flow_features=self.n_flow_features,
            n_task_features=self.n_task_features,
            embedding_dim=self.embedding_dim,
            n_flows=self.n_flows,
            n_tasks=self.n_tasks,
            flow_embed_dim=self.flow_embed_dim,
            task_embed_dim=self.task_embed_dim,
            flow_tower_dims=self.flow_tower_dims,
            task_tower_dims=self.task_tower_dims,
            activation=self.activation,
            use_flow_features=self.use_flow_features,
            use_task_features=self.use_task_features,
        ).to(self.device)

        self._run_training(self.n_epochs)

    def _run_training(
        self,
        n_epochs: int,
        data: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> None:
        """Run the training loop for *n_epochs* on the current model weights.

        Does **not** reinitialise the model — safe to call from both
        :meth:`fit` (after a fresh model is created) and :meth:`partial_fit`
        (warm restart from the current parameters).

        Args:
            data: Optional ``(flow_idx, task_idx, targets)`` tensors to train
                on.  If ``None``, all observed entries in the performance matrix
                are used.
        """
        flow_idx, task_idx, targets = data if data is not None else self._prepare_training_data()
        if len(targets) == 0:
            return

        model = self.model  # capture for closure
        flow_feat_tensor = self._flow_feat_tensor
        task_feat_tensor = self._task_feat_tensor
        loss_fn = self.loss_fn

        def compute_loss(batch: dict) -> torch.Tensor:
            b_fi, b_ti, b_y = batch["fi"], batch["ti"], batch["targets"]
            ff = flow_feat_tensor[b_fi] if flow_feat_tensor is not None else None
            tf = task_feat_tensor[b_ti] if task_feat_tensor is not None else None
            if isinstance(loss_fn, InfoNCELossWithEmbeddings):
                return loss_fn(
                    model.get_flow_embeddings(ff, b_fi),
                    model.get_task_embedding(tf, b_ti),
                )
            return loss_fn(model(ff, tf, b_fi, b_ti), b_y)

        trainer = TorchTrainer(
            n_epochs=n_epochs,
            batch_size=self.batch_size,
            optimizer_factory=self.optimizer_class,
            scheduler_factory=self.scheduler_class,
            device=self.device,
            verbose=self.verbose,
        )
        trainer.run(
            self.model.parameters(),
            compute_loss,
            {"fi": flow_idx, "ti": task_idx, "targets": targets},
        )

    def predict(self, task_index: int) -> np.ndarray:
        """Return predicted scores for all flows on *task_index*.

        Returns:
            Array of shape ``(n_flows,)``.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        with torch.no_grad():
            task_feat = (
                self._task_feat_tensor[task_index : task_index + 1]
                if self._task_feat_tensor is not None
                else None
            )
            task_idx_t = self._all_task_indices[task_index : task_index + 1]

            task_emb = self.model.get_task_embedding(task_feat, task_idx_t)  # (1, d)

            flow_emb = self.model.get_flow_embeddings(
                self._flow_feat_tensor, self._all_flow_indices
            )  # (n_flows, d)

            scores = torch.matmul(flow_emb, task_emb.t()).squeeze(1)
            return scores.cpu().numpy()

    def partial_fit(
        self,
        flow_index: int | list[int],
        task_index: int,
        performance: float | list[float],
    ) -> None:
        """Record new observation(s) and fine-tune for ``n_iter_update`` epochs.

        The model weights are **not** reinitialised — this is a warm restart
        from the current parameters, so only a small number of gradient steps
        are needed to adapt to the new observation.

        When ``update_task_only=True``, fine-tuning uses all observed entries
        for the updated task only (i.e. the full column), rather than all
        observed entries across all tasks.  When ``False`` (default), all
        observed entries across all tasks are used.
        """
        flow_indices = flow_index if isinstance(flow_index, list) else [flow_index]
        performances = performance if isinstance(performance, list) else [performance]

        perf_matrix = self.dataset.require_matrix()
        for fi, p in zip(flow_indices, performances):
            perf_matrix.iloc[fi, task_index] = p

        if self.update_task_only:
            task_col = self.dataset.require_matrix().iloc[:, task_index].values
            observed_mask = ~np.isnan(task_col)
            fi_t = torch.tensor(np.where(observed_mask)[0], dtype=torch.long, device=self.device)
            ti_t = torch.full((int(observed_mask.sum()),), task_index, dtype=torch.long, device=self.device)
            y_t = torch.tensor(task_col[observed_mask], dtype=torch.float32, device=self.device)
            self._run_training(self.n_iter_update, data=(fi_t, ti_t, y_t))
        else:
            self._run_training(self.n_iter_update)

    def suggest(self, task_index: int, predictions=None) -> np.ndarray:
        """Extend base suggest to push flow tower embeddings into EmbeddingMMRAcquisition."""
        if isinstance(self.acquisition_fn, EmbeddingMMRAcquisition) and self.model is not None:
            with torch.no_grad():
                flow_emb = self.model.get_flow_embeddings(
                    self._flow_feat_tensor, self._all_flow_indices
                )
            self.acquisition_fn.set_flow_embeddings(flow_emb.cpu().numpy())
        return super().suggest(task_index, predictions)
