"""InfoNCE (Contrastive) Loss for two-tower models.

InfoNCE is a state-of-the-art contrastive loss widely used in modern recommender systems
and two-tower architectures (e.g., YouTube, Pinterest). It treats each batch as a set of
positive and negative examples, where the correct pairing is positive and all other
in-batch pairings are negatives.

For a batch of (flow, task) pairs, the loss encourages:
- High scores for correct (flow_i, task_i) pairs
- Low scores for incorrect (flow_i, task_j) pairs where i != j

This is equivalent to a cross-entropy loss over a softmax distribution of similarities.
"""

import torch
from torch import nn


class InfoNCELoss(nn.Module):
    """InfoNCE (Contrastive) Loss for two-tower collaborative filtering.

    This loss is computed as:
    L = -log(exp(s_ii / τ) / Σ_j exp(s_ij / τ))

    where s_ij is the similarity between flow_i and task_j, and τ is the temperature.

    The loss encourages the model to assign high similarity to correct (flow, task) pairs
    and low similarity to incorrect pairs. All other pairs in the batch serve as negatives.

    Args:
        temperature: Temperature parameter for scaling similarities (default: 0.07)
                    Lower temperature makes the model more confident in its predictions.
                    Typical values: 0.05-0.1 for recommendation tasks.
        reduction: Reduction method for batch loss - 'mean' or 'sum' (default: 'mean')

    Input:
        predictions: (batch_size,) - predicted scores for correct (flow, task) pairs
                    Note: This is NOT used in InfoNCE. We need embeddings instead.

    Note: This loss requires modification to the training loop to compute the full
          similarity matrix between all flows and tasks in the batch.
    """

    def __init__(self, temperature: float = 0.07, reduction: str = "mean"):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute InfoNCE loss.

        Args:
            predictions: (batch_size,) - predicted scores for the positive pairs
                        These are the dot products between flow_i and task_i embeddings.
            targets: (batch_size,) - ground truth performance values (not used in InfoNCE,
                    but kept for compatibility with the training loop interface)

        Returns:
            loss: Scalar loss value

        Note: Standard InfoNCE requires a full similarity matrix. However, since our
              current training loop only computes positive pair scores, this implementation
              falls back to a simpler pointwise loss. For true InfoNCE, the training loop
              needs to be modified to compute all-pairs similarities in each batch.
        """
        # WARNING: This is a simplified version that doesn't implement true InfoNCE.
        # True InfoNCE requires computing similarities between all (flow, task) pairs
        # in the batch, not just the positive pairs.
        #
        # To implement true InfoNCE, the training loop should:
        # 1. Get flow embeddings: flow_emb = model.get_flow_embeddings(flow_features)
        # 2. Get task embeddings: task_emb = model.get_task_embedding(task_features)
        # 3. Compute similarity matrix: sim_matrix = flow_emb @ task_emb.T / temperature
        # 4. Apply cross-entropy loss where labels are the diagonal (positive pairs)

        # For now, we use a placeholder that encourages high scores for positive pairs
        # This is similar to a ranking loss but not true InfoNCE
        # Negative log-likelihood: we want to maximize predictions (minimize negative predictions)
        loss = -torch.log(torch.sigmoid(predictions / self.temperature))

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")


class InfoNCELossWithEmbeddings(nn.Module):
    """True InfoNCE Loss that operates on embeddings.

    This is the correct implementation of InfoNCE that computes similarities between
    all (flow, task) pairs in a batch and uses cross-entropy over the softmax distribution.

    Args:
        temperature: Temperature parameter for scaling similarities (default: 0.07)
        reduction: Reduction method for batch loss - 'mean' or 'sum' (default: 'mean')

    Usage:
        This requires modifying the training loop to pass embeddings instead of predictions.
        The loss expects:
        - flow_embeddings: (batch_size, embedding_dim)
        - task_embeddings: (batch_size, embedding_dim)

        And computes:
        - similarity_matrix: (batch_size, batch_size) via dot product
        - loss: cross-entropy where labels are the diagonal indices
    """

    def __init__(self, temperature: float = 0.07, reduction: str = "mean"):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.cross_entropy = nn.CrossEntropyLoss(reduction=reduction)

    def forward(
        self,
        flow_embeddings: torch.Tensor,
        task_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Compute InfoNCE loss from embeddings.

        Args:
            flow_embeddings: (batch_size, embedding_dim) - flow embeddings
            task_embeddings: (batch_size, embedding_dim) - task embeddings

        Returns:
            loss: Scalar loss value
        """
        batch_size = flow_embeddings.size(0)

        # Normalize embeddings (optional but recommended)
        flow_embeddings = nn.functional.normalize(flow_embeddings, dim=1)
        task_embeddings = nn.functional.normalize(task_embeddings, dim=1)

        # Compute similarity matrix: (batch_size, batch_size)
        # similarity_matrix[i, j] = dot(flow_i, task_j)
        similarity_matrix = torch.matmul(flow_embeddings, task_embeddings.t())

        # Scale by temperature
        similarity_matrix = similarity_matrix / self.temperature

        # Labels: diagonal elements are the positive pairs
        labels = torch.arange(batch_size, device=similarity_matrix.device)

        # Compute cross-entropy loss
        # This encourages similarity_matrix[i, i] to be high and similarity_matrix[i, j!=i] to be low
        loss = self.cross_entropy(similarity_matrix, labels)

        return loss
