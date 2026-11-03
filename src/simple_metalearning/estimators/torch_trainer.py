"""Reusable training loop for PyTorch models.

``TorchTrainer`` handles the epoch loop, mini-batch shuffling, optimizer /
scheduler management, gradient clipping, and progress logging.  Each call to
:meth:`run` accepts a ``compute_loss`` closure so that any model architecture
can be trained without sub-classing.

Example usage (regression model)
----------------------------------
::

    trainer = TorchTrainer(
        n_epochs=200, batch_size=256,
        optimizer_factory=functools.partial(torch.optim.AdamW, lr=1e-3),
        verbose=True,
    )

    def compute_loss(batch):
        preds = model(batch["x"])
        return F.mse_loss(preds, batch["y"])

    trainer.run(model.parameters(), compute_loss, {"x": X_tensor, "y": y_tensor})

Example usage (task-batched Deep Kernel GP)
--------------------------------------------
Each "sample" is a task index; N = number of training tasks::

    trainer.run(
        params,
        compute_dkn_loss,   # closure over dkn, gp, mll
        {"task_idx": torch.arange(n_train_tasks, device=device)},
    )
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional

import torch
from torch import nn


class TorchTrainer:
    """Generic training loop.

    Args:
        n_epochs: Default number of epochs.
        batch_size: Mini-batch size (number of *samples* per gradient step,
            where a "sample" is one row of each data tensor).
        optimizer_factory: Callable that takes an iterable of parameters and
            returns a :class:`torch.optim.Optimizer`.  Use
            ``functools.partial(torch.optim.AdamW, lr=1e-3)`` or similar.
        scheduler_factory: Optional callable that takes an optimizer and
            returns a learning-rate scheduler.  The scheduler is stepped once
            per epoch with ``scheduler.step(epoch)``.
        device: Device on which ``torch.randperm`` is created.  Data tensors
            must already reside on this device before calling :meth:`run`.
        verbose: Show a tqdm progress bar.
        grad_clip: If set, clip gradient norms to this value before each
            optimizer step.
    """

    def __init__(
        self,
        n_epochs: int,
        batch_size: int,
        optimizer_factory: Callable,
        scheduler_factory: Optional[Callable] = None,
        device: Optional[torch.device] = None,
        verbose: bool = False,
        grad_clip: Optional[float] = None,
    ) -> None:
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.device = device or torch.device("cpu")
        self.verbose = verbose
        self.grad_clip = grad_clip

    def run(
        self,
        parameters: Iterable[nn.Parameter],
        compute_loss: Callable[[dict[str, torch.Tensor]], torch.Tensor],
        data: dict[str, torch.Tensor],
        n_epochs: Optional[int] = None,
    ) -> list[float]:
        """Train for *n_epochs* epochs.

        Each epoch, the *N* samples in *data* are randomly permuted and
        split into mini-batches of size ``batch_size``.  For each mini-batch,
        ``compute_loss(batch)`` is called where *batch* is a dict containing
        the corresponding row-slices of every tensor in *data*.

        A fresh optimizer (and optional scheduler) is created at the start of
        each call to :meth:`run`.  This means calling :meth:`run` twice is
        equivalent to two independent training runs, not a continuation.  Use
        a smaller *n_epochs* for fine-tuning.

        Args:
            parameters: Model parameters to optimise (passed directly to
                ``optimizer_factory``).
            compute_loss: Callable that takes a batch dict and returns a
                scalar loss tensor.
            data: Dict of tensors.  All values must have the same leading
                dimension *N* (the number of samples).
            n_epochs: Override :attr:`n_epochs` for this call.

        Returns:
            List of average per-batch losses, one per epoch.
        """
        # Create a concrete list so we can pass it to the optimizer multiple times
        params = list(parameters)
        optimizer = self.optimizer_factory(params)
        scheduler = (
            self.scheduler_factory(optimizer) if self.scheduler_factory is not None else None
        )

        n = len(next(iter(data.values())))
        epochs = n_epochs if n_epochs is not None else self.n_epochs
        epoch_losses: list[float] = []

        from tqdm import tqdm

        bar = tqdm(range(epochs), disable=not self.verbose, desc="Training", leave=False)
        for epoch in bar:
            perm = torch.randperm(n, device=self.device)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                batch = {k: v[idx] for k, v in data.items()}

                optimizer.zero_grad()
                loss = compute_loss(batch)
                loss.backward()

                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(params, self.grad_clip)

                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            if scheduler is not None:
                scheduler.step(epoch)

            avg = epoch_loss / max(n_batches, 1)
            epoch_losses.append(avg)

            if self.verbose:
                bar.set_postfix(
                    loss=f"{avg:.6f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )

        return epoch_losses
