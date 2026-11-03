# Loss Functions for Two-Tower Estimators

This directory contains custom loss functions for training two-tower collaborative filtering models.

## InfoNCE Loss (Contrastive Loss)

**File:** `info_nce.py`

### Overview
InfoNCE (Information Noise Contrastive Estimation) is a state-of-the-art contrastive loss widely used in modern two-tower recommender systems. It's the foundation for systems used by:
- YouTube recommendations
- Pinterest visual search
- Spotify music recommendations
- Many modern retrieval systems

### How It Works

For a batch of (flow, task) pairs:
1. Compute embeddings for all flows and tasks in the batch
2. Calculate similarity matrix: `S[i,j] = dot(flow_i, task_j) / temperature`
3. Apply cross-entropy loss where positive pairs (i,i) should have high similarity
4. All other pairs (i,j) where i≠j serve as in-batch negatives

The loss encourages:
- **High similarity** for correct (flow_i, task_i) pairs
- **Low similarity** for incorrect (flow_i, task_j) pairs where i ≠ j

### Key Features

**Advantages:**
- **State-of-the-art performance** for ranking/retrieval tasks
- **No need for explicit negative sampling** - uses in-batch negatives
- **Scales well** with batch size (larger batches = more negatives)
- **Works naturally** with two-tower architectures and dot product similarity

**Hyperparameters:**
- `temperature` (default: 0.07): Controls the sharpness of the softmax distribution
  - Lower (0.05): More confident, sharper distinctions
  - Higher (0.1): Softer, more exploratory
  - Typical range: 0.05-0.1 for recommendation tasks

### Usage

#### In Config File:
```yaml
# conf/estimator/two_tower.yaml
loss_fn:
  _target_: simple_metalearning.estimators.losses.info_nce.InfoNCELossWithEmbeddings
  temperature: 0.07
  reduction: mean
```

#### Or use the pre-configured version:
```bash
python -m simple_metalearning.main estimator=two_tower_infonce
```

### When to Use InfoNCE vs MSE

**Use InfoNCE when:**
- You care about **ranking** (which flow is best?) more than exact performance prediction
- You have **sufficient batch size** (>=32 recommended, larger is better)
- You want to learn **semantic similarity** between flows and tasks
- Your evaluation metrics are rank-based (e.g., NDCG, MRR, precision@k)

**Use MSE when:**
- You need **exact performance values** (not just rankings)
- You have **small batches** (InfoNCE less effective with few negatives)
- You want to minimize **absolute prediction error**
- Your evaluation metrics are pointwise (e.g., MAE, RMSE)

### Implementation Details

The `InfoNCELossWithEmbeddings` class:
1. Takes flow and task embeddings as input (not predictions)
2. Normalizes embeddings to unit length (optional but recommended)
3. Computes full similarity matrix via matrix multiplication
4. Applies temperature scaling
5. Uses cross-entropy with diagonal labels

The training loop in `two_tower.py` automatically detects InfoNCE loss and:
- Extracts embeddings from the model
- Passes them to the loss function
- Skips the prediction computation step

### Mathematical Formulation

For a batch of size N:

```
flow_emb = normalize(FlowTower(flow_features))     # (N, d)
task_emb = normalize(TaskTower(task_features))     # (N, d)

S = (flow_emb @ task_emb.T) / τ                    # (N, N) similarity matrix

L = -log(exp(S[i,i]) / Σ_j exp(S[i,j]))           # Cross-entropy loss
```

Where:
- `τ` is the temperature
- `S[i,i]` are the positive pairs (diagonal)
- `S[i,j]` for j≠i are the negative pairs (off-diagonal)

### References

1. **Original Paper:** [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748)
2. **SimCLR:** [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
3. **Two-Tower Applications:**
   - [Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations](https://research.google/pubs/pub48840/) (YouTube)
   - [Embedding-based Retrieval in Facebook Search](https://arxiv.org/abs/2006.11632)
