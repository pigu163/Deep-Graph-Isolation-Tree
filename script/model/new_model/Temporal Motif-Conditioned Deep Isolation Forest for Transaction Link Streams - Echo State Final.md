# Temporal Motif-Conditioned Deep Isolation Forest with Contractive Transaction Graph Echo-State Reservoirs

> **Implementation specification for Codex and other coding agents**
>
> Status: final implementation specification
>
> Updated: 2026-08-06
>
> Target task: transaction-level unsupervised anomaly detection on large temporal transaction graphs

---

## 0. Executive summary

This project implements an optimization-free, transaction-level anomaly detector for large temporal transaction link streams.

Each transaction is one event:

\[
e_i = (t_i, u_i, v_i, x_i)
\]

where:

- `t_i`: timestamp;
- `u_i`: sender node id;
- `v_i`: receiver node id;
- `x_i`: transaction edge feature vector;
- optional `y_i`: transaction-level label used only for evaluation, never for fitting.

The method combines two ideas:

1. **DIF-style random representation ensemble**
   - Build several independently randomized transaction graph echo-state reservoirs.
   - Never train the reservoir, recurrent weights, input weights, or transaction projection.
   - Enforce a strict contraction condition for every reservoir member at initialization and model loading.
   - Use the resulting causal fading-memory transaction representations as the spaces in which isolation trees are built.

2. **Temporal motif-conditioned isolation split**
   - Do not split by a random embedding coordinate.
   - At each tree node, randomly choose one AML-relevant temporal motif.
   - Split transactions according to a scalar response that combines:
     - deviation of the current transaction embedding from the motif-conditioned historical context embedding;
     - structural increment caused by the current transaction for that motif.

The complete pipeline is:

```text
existing link-stream data pipeline
    -> transactions sorted by (timestamp, event_id)
    -> past-only temporal graph / motif state
    -> R frozen contractive transaction graph echo-state reservoirs
    -> one transaction embedding per event per reservoir
    -> motif-conditioned context and motif increment
    -> motif-conditioned isolation trees
    -> mean path length over all trees
    -> one transaction anomaly score
```

The detection target is always the transaction. Sender and receiver states are internal historical context; the model does not perform node classification and does not output sender-level or receiver-level anomaly labels.

The fitted model must output exactly one anomaly score per transaction.

---

## 1. Non-negotiable design constraints

The implementation MUST satisfy all of the following.

1. **Transaction-level output**
   - One input transaction event produces one embedding per reservoir and one final anomaly score.
   - Sender and receiver states are latent historical context only.
   - Do not turn the task into node classification, account scoring, graph classification, or link prediction.

2. **Chronological mini-batch causality**
   - Sort events deterministically by `(timestamp, event_id)` and form contiguous mini-batches without shuffling.
   - Every event in a mini-batch reads the same pre-batch state; node, graph, motif, and context updates are committed only after the complete mini-batch is scored.
   - Equal timestamps require no special grouping. The mini-batch is the discrete reservoir time step.

3. **No neural training**
   - No labels.
   - No losses.
   - No optimizer.
   - No backpropagation.
   - Reservoir weights and transaction projection weights are initialized once and frozen.

4. **Echo-state reservoir, not frozen TGN**
   - Do not implement TGN memory, learned GRU/LSTM cells, attention, or trainable message functions.
   - Use an event-driven dynamic graph echo-state reservoir specialized for directed transaction link streams.
   - Each reservoir member must pass an explicit contraction verification.

5. **DIF-style ensemble diversity**
   - Use `R` independently initialized reservoir members.
   - Each member owns an independent node-state store and independent frozen weights.
   - Do not use CERE modulation for recurrent weights in the main method because every member must be independently verified to satisfy the contraction bound.

6. **New tree split**
   - Do not use the standard IF/DIF split `h[k] < theta` as the main method.
   - Do not use an arbitrary random projection as the main split.
   - Use the motif-conditioned scalar response defined in Section 9.

7. **Scalability**
   - The full graph must never be materialized as a dense adjacency matrix.
   - Use sparse streaming state, compact hash maps, bounded temporal neighborhoods, and isolation-tree subsampling.
   - Reservoir state updates touch only nodes participating in the current mini-batch.

8. **Fit/scoring separation**
   - Fitting builds reservoir-specific transaction statistics and trees.
   - Scoring traverses fitted trees and updates online state only after scoring.

9. **Use the existing data pipeline**
   - Do not reimplement dataset classes, file readers, raw preprocessing, or link-stream construction.
   - Integrate with the existing chronological transaction iterator and existing event tensors.

---

## 2. Problem definition

### 2.1 Data model

The dataset contains `N` rows sorted by time:

```python
@dataclass(frozen=True)
class TransactionEvent:
    event_id: int
    timestamp: float
    sender: int
    receiver: int
    edge_features: torch.Tensor  # [d_e]
    label: int | None = None      # evaluation only
```

Tensor batch representation:

```python
src:        LongTensor[B]
ts:         FloatTensor[B]
dst:        LongTensor[B]
edge_feat:  FloatTensor[B, d_e]
event_id:   LongTensor[B]
label:      LongTensor[B] | None
```

### 2.2 Detection objective

Given a transaction event and its past graph state:

\[
(e_i, \mathcal{G}_{<b(i)})
\]
where `b(i)` is the deterministic chronological mini-batch containing event
`i`, and `G_<b(i)` is the committed stream prefix before that mini-batch.


produce:

\[
S(e_i) \in [0,1]
\]

where a larger score means more anomalous.

The ground-truth label, when available, is used only for metrics such as ROC-AUC, PR-AUC, Precision@K, and Recall@K.

---

## 3. High-level components

The existing dataset and link-stream pipeline are out of scope and must remain unchanged.

Replace the previous TGN encoder subsystem with the following conceptual components:

```text
ContractiveReservoirWeightFactory
TransactionGraphEchoStateReservoir
TransactionGraphReservoirEnsemble
ReservoirNodeStateStore
ContractionVerifier
```

Keep the existing temporal state, motif, context, tree, forest, model, evaluation, CLI, and serialization modules wherever possible.

The coding agent may choose exact class and file names to match the existing repository. The implementation contract is behavioral, not tied to a mandatory API shape.

Required responsibilities:

```text
ContractiveReservoirWeightFactory
    initialize recurrent matrices using selectable layouts
    rescale every recurrent matrix to a strict operator-norm target below 1
    freeze all generated tensors

ReservoirNodeStateStore
    store one state vector per active node and per reservoir member
    return the zero vector for unseen nodes
    support mini-batch read and mean-commit semantics

TransactionGraphEchoStateReservoir
    read pre-batch sender and receiver states
    construct one ordered transaction embedding before state update
    compute node-separable contractive endpoint candidates
    mean repeated-node candidates before commit
    expose pending updates without mutating the state store

TransactionGraphReservoirEnsemble
    run R independent reservoir members
    return one transaction embedding per event per member

ContractionVerifier
    compute or tightly estimate the operator 2-norm
    verify the analytical contraction factor
    reject invalid initialized or loaded reservoirs
```

---

## 4. Past-only temporal state and sampling

### 4.1 Required state

Maintain sparse streaming state:

```python
last_event_time: Dict[node_id, float]
last_pair_time: Dict[(sender, receiver), float]
recent_out_edges: Dict[sender, Deque[EdgeRecord]]
recent_in_edges: Dict[receiver, Deque[EdgeRecord]]
recent_pair_edges: Dict[(sender, receiver), Deque[EdgeRecord]]
node_state_per_reservoir: List[Dict[node_id, Tensor[d_s]]]
```

Each `EdgeRecord` should contain at least:

```python
@dataclass
class EdgeRecord:
    event_id: int
    timestamp: float
    sender: int
    receiver: int
    edge_features: torch.Tensor  # [d_e]
    event_embedding: torch.Tensor | None  # [d_h], per reservoir member when needed
```

Do not keep unlimited history in RAM. Enforce at least one bound:

- fixed time window `W`;
- maximum recent edges per node `K_node`;
- maximum recent edges per pair `K_pair`.

Recommended first implementation:

```yaml
temporal_window_seconds: 604800   # 7 days
max_recent_edges_per_node: 128
max_recent_edges_per_pair: 32
```

### 4.2 Past-only sampler

For target event `e_i`, sample a temporal ego-subgraph centered on `(u_i, v_i)` using only historical edges.

Minimal sampling rule:

1. collect recent outgoing edges of `u_i`;
2. collect recent incoming edges of `v_i`;
3. collect recent interactions of pair `(u_i, v_i)`;
4. optionally collect one-hop edges touching the intermediate nodes found above;
5. remove duplicates;
6. keep only events with timestamp `< t_i`;
7. sort by `(timestamp, event_id)`;
8. truncate to `max_subgraph_edges`.

Interface:

```python
@dataclass
class TemporalSubgraph:
    target_event_id: int
    node_ids: LongTensor[N_g]
    src: LongTensor[E_g]
    dst: LongTensor[E_g]
    ts: FloatTensor[E_g]
    edge_feat: FloatTensor[E_g, d_e]
    event_ids: LongTensor[E_g]

class PastOnlyTemporalSampler:
    def sample(self, event: TransactionEvent, state: TemporalStateStore) -> TemporalSubgraph:
        ...
```

The target transaction is not included in the historical edge list. It is supplied separately to the transaction reservoir.

---

## 5. Contractive event-driven transaction graph echo-state reservoir

### 5.1 Design goal

Implement a training-free dynamic graph reservoir for a directed transaction link stream.

The reservoir must:

- maintain one persistent hidden state per node;
- update only nodes participating in the current mini-batch;
- use sender and receiver roles asymmetrically;
- incorporate transaction edge features and elapsed-time features;
- produce the transaction embedding from pre-batch states;
- update states only after the transaction has been encoded and scored;
- satisfy an explicit strict contraction condition;
- support multiple frozen recurrent-matrix initialization layouts.

This is not a TGN implementation. Do not include learned message functions, learned recurrent cells, temporal attention, loss functions, or optimizers.

### 5.2 State and dimensions

For reservoir member `r`, every node `v` owns a state:

\[
s_v^{(r)} \in [-1,1]^{d_s}.
\]

Use:

```text
d_e     preprocessed transaction feature dimension
d_t     fixed time encoding dimension
d_s     node reservoir state dimension
d_h     transaction embedding dimension
R       number of independent reservoir members
B       mini-batch event count
```

Recommended defaults:

```yaml
reservoir_state_dim: 64
transaction_embedding_dim: 64
time_dim: 16
num_reservoirs: 8
```

For a mini-batch:

```text
src_state:              [B, d_s]
dst_state:              [B, d_s]
src_time_feature:       [B, d_t]
dst_time_feature:       [B, d_t]
transaction_features:   [B, d_e]
transaction_input:      [B, d_a]
transaction_embedding:  [B, d_h]
```

### 5.3 New-node handling

An unseen sender or receiver receives the deterministic initial state:

\[
s_v^{(r)} = \mathbf{0}.
\]

No learned node embedding table is allowed.

For a new node:

```text
state               = zeros(d_s)
last_event_time     = missing
is_new_node         = 1
```

Use a deterministic cold-start time feature. The implementation may use either:

- a dedicated `is_new_node` indicator together with a zero time gap; or
- a bounded configured cold-start time gap together with the indicator.

New-node status is only an input feature. It must not directly force a high anomaly score.

### 5.4 Fixed time encoding

Use the existing fixed non-trainable sinusoidal or log-sinusoidal time encoding. No learned time encoder is required.

For `delta_t >= 0`:

\[
\phi(\Delta t)
=
[\sin(\omega_1\log(1+\Delta t)),\cos(\omega_1\log(1+\Delta t)),\ldots].
\]

Clamp negative time gaps to zero and assert chronological processing.

### 5.5 Transaction and endpoint inputs

For event `e_i = (t_i, u_i, v_i, x_i)`, construct the bounded event input:

\[
a_i = [\widetilde{x}_i \Vert \phi(\Delta t_{u_i}) \Vert
\phi(\Delta t_{v_i}) \Vert \mathbf{1}[u_i\text{ new}] \Vert
\mathbf{1}[v_i\text{ new}]] \in \mathbb{R}^{d_a}.
\]

The endpoint input adds one ordered role bit:

\[
a_{i,u} = [a_i \Vert 0], \qquad a_{i,v} = [a_i \Vert 1].
\]

The implementation MUST keep recurrent inputs exogenous to the reservoir state.
Do not feed another node's reservoir state, a reservoir-derived context, or the
current transaction embedding into the recurrent input. Historical topology,
motif features, and pair context remain available to the transaction embedding
and isolation forest.

### 5.6 Endpoint-wise reservoir update

For reservoir member `r`, every node owns one state
`s_v^(r) in [-1,1]^{d_s}`. All events in a chronological mini-batch read the
same pre-batch node state. For each endpoint occurrence `(i,v)`, compute:

\[
\widehat{s}_{i,v}
=
\tanh(W_{in}a_{i,v}+W_{res}s_v^-+b),
\]

\[
\boxed{
\widetilde{s}_{i,v}
=
(1-\gamma)s_v^-+\gamma\widehat{s}_{i,v}
}.
\]

The frozen matrices have dimensions:

```text
W_res:    [d_s, d_s]
W_in:     [d_s, d_a + 1]
b:        [d_s]
```

A mini-batch of `B` transactions is flattened into `2B` endpoint occurrences:

```text
old_endpoint_state: [R, 2B, d_s]
endpoint_input:     [R, 2B, d_a + 1]
endpoint_candidate: [R, 2B, d_s]
```

This is the dense CUDA ESN bank. Reservoir members are stacked only for batched
matrix multiplication; weights and node-state stores remain independent.

If node `v` occurs `m_v` times in the mini-batch, commit the arithmetic mean:

\[
\boxed{
s_v^+ = \frac{1}{m_v}\sum_{i\in I_v}\widetilde{s}_{i,v}
}.
\]

An inactive node keeps its previous state. No timestamp grouping, edge coloring,
or sequential state update inside a mini-batch is required.

### 5.7 Strict contraction guarantee

For a fixed endpoint input `a`, define:

\[
F_a(s)=(1-\gamma)s+\gamma\tanh(W_{in}a+W_{res}s+b).
\]

Because element-wise `tanh` is 1-Lipschitz in the Euclidean norm:

\[
\|F_a(s)-F_a(s')\|_2
\leq
\left[(1-\gamma)+\gamma\|W_{res}\|_2\right]\|s-s'\|_2.
\]

Define:

\[
\boxed{\kappa=(1-\gamma)+\gamma\|W_{res}\|_2}.
\]

The implementation MUST enforce `||W_res||_2 <= eta < 1`; therefore
`kappa < 1`. For repeated-node mean aggregation:

\[
\begin{aligned}
\|T_v(s)-T_v(s')\|_2
&=\left\|\frac{1}{m_v}\sum_{i\in I_v}
(F_{a_{i,v}}(s)-F_{a_{i,v}}(s'))\right\|_2\\
&\leq\frac{1}{m_v}\sum_{i\in I_v}
\kappa\|s-s'\|_2\\
&=\kappa\|s-s'\|_2.
\end{aligned}
\]

Thus every active node update remains strictly contractive for arbitrary
mini-batches, including repeated senders, repeated receivers, and star-shaped
batches. There is no cross-node recurrent-state path, so one node's initial-state
perturbation cannot be copied into multiple neighboring node states.

### 5.8 Scope of the guarantee

The full sparse graph-state map is non-expansive because inactive nodes are
unchanged. Every active node contracts by at most `kappa` on each mini-batch in
which it participates. Consequently, for a node active in `n` future batches:

\[
\|s_{v,n}-s'_{v,n}\|_2
\leq
\kappa^n\|s_{v,0}-s'_{v,0}\|_2.
\]

This is a sufficient node-wise echo-state/fading-memory guarantee indexed by
chronological mini-batch activations. It is not a claim of infinite memory,
optimal memory capacity, or anomaly-detection accuracy.

### 5.9 Guaranteed weight normalization

The strict guarantee uses the operator 2-norm, equal to the largest singular value:

\[
\|W_{res}\|_2 = \sigma_{max}(W_{res}).
\]

Do not rely only on the spectral radius for the formal guarantee. For a non-normal matrix, `rho(W) < 1` does not imply `||W||_2 < 1`.

For every raw recurrent matrix `W_raw`, construct:

\[
W_{res}
=
\eta
\frac{W_{raw}}
{\max(\sigma_{max}(W_{raw}),\epsilon)}.
\]

Then verify numerically:

```text
operator_norm(W_res) <= eta + tolerance
eta < 1
kappa = (1 - leak_rate) + leak_rate * operator_norm(W_res)
kappa < 1
```

This verification is mandatory:

- immediately after initialization;
- after deserialization;
- before fitting;
- before scoring a loaded model.

If verification fails, raise an error. Do not silently renormalize loaded model weights unless explicitly running a migration utility.

### 5.10 Reservoir recurrent-matrix layouts

Support selectable raw recurrent-matrix layouts inspired by the reservoir designs analyzed in the 2024 GESN reservoir-design study.

All layouts MUST be followed by the operator-norm normalization in Section 5.9. The layout controls structure and spectrum before final contraction scaling; no layout bypasses the guarantee.

Required layouts:

```text
random_dense_uniform
random_dense_gaussian
orthogonal
symmetric
antisymmetric
ring
sparse_random
block_diagonal_grouped
```

Implementation requirements:

#### `random_dense_uniform`

- Draw independent zero-mean uniform entries.
- Scale variance according to matrix width before final norm normalization.

#### `random_dense_gaussian`

- Draw independent zero-mean Gaussian entries.
- Scale variance according to matrix width before final norm normalization.

#### `orthogonal`

- Draw a random square matrix.
- Obtain an orthogonal factor using QR decomposition.
- Resolve QR sign ambiguity deterministically from the diagonal of `R`.
- Scale by the configured operator-norm target.

Orthogonal reservoirs are important because orthogonal recurrent matrices are associated with improved memory capacity in ESNs.

#### `symmetric`

- Construct `W_raw = (M + M.T) / 2`.
- Then apply final operator-norm normalization.

#### `antisymmetric`

- Construct `W_raw = (M - M.T) / 2`.
- Then apply final operator-norm normalization.
- Pure antisymmetry may be combined with a small diagonal or identity component only if the resulting full matrix is subsequently re-normalized and recorded in metadata.

#### `ring`

- Construct a cyclic permutation / shift-register matrix.
- Optionally apply deterministic signed entries.
- Scale to the configured target norm.

#### `sparse_random`

- Build a sparse matrix with configured density or fixed fan-in per reservoir unit.
- Ensure no unintended all-zero rows unless explicitly allowed.
- Compute or tightly estimate the largest singular value and normalize.

#### `block_diagonal_grouped`

- Partition reservoir units into independent groups.
- Initialize each block using a configured base layout.
- Build the block-diagonal matrix.
- Normalize the full matrix and verify the full operator norm.

The reservoir layout is an experimental configuration. The main anomaly detector, transaction representation contract, motif-conditioned split, and scoring rule remain identical across layouts.

### 5.11 Transaction embedding

The sample representation is a transaction embedding computed before the same event updates node states.

Construct:

\[
q_i
=
[
s_u^-
\Vert
s_v^-
\Vert
a_i
].
\]

Then compute:

\[
h_i
=
\operatorname{LayerNorm}
\left(
\tanh(W_{event}q_i+b_{event})
\right).
\]

Requirements:

- `W_event` and `b_event` are randomly initialized and frozen;
- `W_event` is not recurrent and does not require a contraction constraint;
- use non-affine LayerNorm;
- preserve sender/receiver order;
- return exactly one transaction embedding per event;
- do not output node embeddings as anomaly samples.

### 5.12 Causal mini-batch semantics

For every mini-batch:

```text
1. read pre-batch sender and receiver states
2. construct every transaction input
3. compute every transaction embedding
4. compute motif contexts and motif increments
5. cache features or score every transaction
6. compute pending reservoir state candidates
7. aggregate multiple pending updates for the same node
8. commit reservoir node states
9. commit graph, timestamp, motif, and context state
```

Every event in the same mini-batch must read the same pre-batch state.

If one node participates in multiple transactions in the mini-batch, compute
all candidates from its same old state, take their arithmetic mean, and apply
one final state commit for that node.

### 5.13 Frozen-parameter requirements

All reservoir and transaction-projection tensors must be frozen.

```text
requires_grad == False for every parameter
no optimizer object
no loss computation
no backward call
no learned readout
```

The Isolation Forest consumes frozen transaction representations directly; unlike standard reservoir-computing supervised applications, this method does not train a linear readout.

---

## 6. Contractive transaction reservoir ensemble

Use `R` independently initialized frozen transaction graph echo-state reservoirs:

\[
\mathcal R^{(1)},\ldots,\mathcal R^{(R)}.
\]

For transaction `e_i`:

\[
h_i^{(r)}
=
\mathcal R^{(r)}(e_i,\mathcal G_{<b(i)}).
\]

Output for one mini-batch:

```text
H: [R, B, d_h]
```

Each reservoir member owns:

- independent frozen input weights;
- an independently initialized and independently verified recurrent matrix;
- independent frozen transaction-projection weights;
- an independent node-state store;
- its own contraction report and random seed.

For every member `r`:

\[
\|W_{res}^{(r)}\|_2 \le \eta_r < 1,
\]

\[
\kappa_r
=
(1-\gamma_r)+\gamma_r\|W_{res}^{(r)}\|_2
<1.
\]

Recommended defaults:

```yaml
num_reservoirs: 8
trees_per_reservoir: 25
subsample_size: 256
max_tree_depth: 8
```

Total trees:

\[
T_{total}=R\times T.
\]

Do not share recurrent states across reservoir members. Do not apply unverified recurrent perturbations after normalization.

---

## 7. Temporal motifs

Use exactly four motifs in the main method.

```python
class MotifType(str, Enum):
    FAN_OUT = "fan_out"
    FAN_IN = "fan_in"
    RAPID_CHAIN = "rapid_chain"
    CYCLE = "cycle"
```

All motif computations are past-only and local.

### 7.1 Fan-out

Measures whether sender `u_i` is distributing funds to many distinct receivers within the motif window.

Historical count:

\[
M_{out}^{before}(e_i)
= |\{v: (u_i,v,t)\in\mathcal{G},\ t_i-W_m \le t < t_i\}|
\]

Increment:

\[
\Delta M_i^{out}
= \mathbb{1}[v_i \text{ is a new receiver for } u_i \text{ in the window}]
\]

Use a weighted version if desired:

\[
\Delta M_i^{out}
= \log(1 + 1 + \text{recent distinct receiver growth})
\]

### 7.2 Fan-in

Measures whether receiver `v_i` is collecting funds from many distinct senders.

\[
\Delta M_i^{in}
= \mathbb{1}[u_i \text{ is a new sender to } v_i \text{ in the window}]
\]

### 7.3 Rapid chain

Measures rapid pass-through behavior.

For event `u_i -> v_i` at time `t_i`, inspect recent outgoing edges of `v_i`:

\[
(v_i \to w, t_j), \quad t_i < t_j \le t_i + W_{chain}
\]

Past-only online scoring cannot see future edges. Therefore define the causal version using prior incoming activity of `u_i`:

```text
w -> u_i shortly before u_i -> v_i
```

Causal increment:

\[
\Delta M_i^{chain}
= \sum_{(w,u_i,t_j)} \exp[-\gamma(t_i-t_j)]
\]

for recent incoming edges to `u_i`.

This identifies an event that continues a recently received flow.

### 7.4 Cycle closure

Measures whether current edge `u_i -> v_i` closes a short directed cycle in the historical graph.

Minimal 2-step closure:

```text
v_i -> w -> u_i  exists recently
current event: u_i -> v_i
```

Define:

\[
\Delta M_i^{cycle}
= \sum_w
\mathbb{1}[(v_i,w)\in G_{<b(i)}]
\mathbb{1}[(w,u_i)\in G_{<b(i)}]
\cdot \text{time_decay}
\]

Bound neighbor scans by `K_motif_neighbors`.

---

## 8. Motif-conditioned historical context

### 8.1 Context event set

For each target event and motif, select past event embeddings relevant to that motif.

```text
fan_out context:
    past outgoing events of current sender

fan_in context:
    past incoming events of current receiver

rapid_chain context:
    recent incoming events of current sender

cycle context:
    recent events on paths v -> w -> u and direct pair history
```

For reservoir member `r`, define the context set:

\[
\mathcal{C}_i^{(m)} \subset \{h_j^{(r)}: t_j < t_i\}
\]

### 8.2 Time-decayed context embedding

\[
c_i^{(r,m)} =
\frac{
\sum_{j\in\mathcal C_i^{(m)}}
\exp[-\beta_m(t_i-t_j)]h_j^{(r)}
}{
\sum_{j\in\mathcal C_i^{(m)}}
\exp[-\beta_m(t_i-t_j)] + \epsilon
}
\]

Shape:

```text
context_embedding: [d_h]
```

### 8.3 Cold start

If the context set is empty:

```python
context_embedding = torch.zeros(d_h)
context_count = 0
context_available = False
```

Do not allow cold start alone to dominate the split. Apply confidence shrinkage:

\[
\rho_i^{(m)} = \frac{n_i^{(m)}}{n_i^{(m)} + \alpha}
\]

\[
\tilde c_i^{(r,m)} = \rho_i^{(m)} c_i^{(r,m)} + (1-\rho_i^{(m)}) c_{global}^{(r)}
\]

Recommended:

```yaml
context_shrinkage_alpha: 5.0
```

Maintain `c_global^(r)` as a streaming time-decayed mean of past embeddings for each reservoir member.

### 8.4 Context storage

For scalability, do not store every historical embedding forever.

Use bounded deques of recent embeddings per node and pair, or maintain exponentially weighted sums:

```python
weighted_sum: Tensor[d_h]
weight_total: float
count: int
last_update_time: float
```

For rapid-chain and cycle contexts, recent bounded adjacency deques are still required.

---

## 9. Final isolation split

### 9.1 Motif-conditioned response

At each tree node, randomly choose one motif `m` from the four motif types.

For event `i`, reservoir member `r`, and motif `m`, compute:

\[
d_i^{(r,m)}
=
\frac{\|h_i^{(r)}-\tilde c_i^{(r,m)}\|_2}{\sqrt{d_h}}
\]

Normalize motif increment within the tree-building sample using a robust fixed transform:

\[
g_i^{(m)} = \log(1 + \max(0, \Delta M_i^{(m)}))
\]

Final response:

\[
\boxed{
q_i^{(r,m)}
=
d_i^{(r,m)} + \lambda_m g_i^{(m)}
}
\]

This scalar is the only value used for the node split.

### 9.2 Random threshold

Given current node sample indices `A`:

\[
q_{min} = \min_{i\in A} q_i^{(r,m)},\quad
q_{max} = \max_{i\in A} q_i^{(r,m)}
\]

If `q_max - q_min < split_eps`, return a leaf.

Otherwise:

\[
\theta \sim U(q_{min}, q_{max})
\]

Split:

\[
A_L = \{i \in A: q_i^{(r,m)} < \theta\}
\]

\[
A_R = \{i \in A: q_i^{(r,m)} \ge \theta\}
\]

If either side is empty, retry up to `max_split_retries`; then return a leaf.

### 9.3 Tree node storage

Each internal node stores:

```python
@dataclass
class InternalNode:
    motif_type: MotifType
    threshold: float
    lambda_motif: float
    left: "TreeNode"
    right: "TreeNode"
```

Do not store a random feature index or random projection vector.

### 9.4 Why the split is different

```text
IF:
    raw_feature[k] < threshold

DIF:
    random_embedding[k] < threshold

This method:
    distance(current random temporal event embedding,
             motif-conditioned past behavior)
    + motif structural increment
    < threshold
```

The split is event-specific, temporal, structural, and causal.

---

## 10. Fitting procedure

### 10.1 Data split

Use a chronological split supplied by the existing data pipeline.

```text
fit period        -> build reservoir transaction representations, contexts, and trees
validation period -> select engineering hyperparameters under the chosen protocol
holdout test      -> final metrics
```

Never randomly shuffle transactions across time.

### 10.2 Representation pass

For each reservoir member `r`:

1. initialize and verify frozen weights;
2. reset its node reservoir state;
3. reset motif/context state;
4. stream through the fit period in timestamp order;
5. for every mini-batch, read pre-batch states;
6. compute one transaction embedding per event;
7. compute four motif-conditioned context distances;
8. compute four motif increments;
9. persist compact tree features;
10. commit reservoir and temporal state only after the read-only phase.

Persist only what the trees need:

```text
EventTreeFeatures
    event_id
    context_distance: [4]
    motif_increment:  [4]
```

The full transaction embedding need not be persisted after all required context distances are computed, unless enabled for debugging or analysis.

### 10.3 Tree building

For each reservoir member `r`, build `T` trees.

Each tree:

1. samples `psi` transaction indices without replacement;
2. recursively builds nodes;
3. randomly chooses one motif at every node;
4. computes the existing motif-conditioned scalar response;
5. samples the threshold uniformly between the node minimum and maximum;
6. recurses until the existing stopping conditions are met.

The tree algorithm and split definition remain unchanged.

### 10.4 Contraction audit during fitting

Before the representation pass, record for every reservoir member:

```text
layout
seed
leak_rate
configured_target_operator_norm
measured_operator_norm
analytical_kappa
verification_tolerance
verification_passed
```

Fitting must fail if any reservoir member has `verification_passed == false`.

---

## 11. Inference and online detection

For each chronological mini-batch:

1. read pre-batch sender and receiver states for all reservoir members;
2. run the endpoint-wise ESN bank and ordered pair projection;
3. compute four motif-conditioned distances and increments per transaction;
4. traverse the assigned trees and output one score per transaction;
5. only after the complete mini-batch is scored, commit:
   - mean-aggregated reservoir node-state updates;
   - temporal adjacency and timestamps;
   - motif counters;
   - context statistics.

The detection target remains the transaction. Node states are never directly converted into account labels.

### 11.1 Tree traversal

Tree traversal is unchanged and uses the stored motif type, threshold, and motif coefficient.

### 11.2 Path length

Use the existing Isolation Forest external-node correction and path-length definition.

### 11.3 Anomaly score

Average path length over all reservoir-specific trees:

\[
\bar\ell(e)
=
\frac{1}{RT}
\sum_{r=1}^{R}\sum_{j=1}^{T}\ell_{r,j}(e).
\]

Final score:

\[
\boxed{
S(e)=2^{-\bar\ell(e)/c(\psi)}
}.
\]

Higher score means more anomalous.

---

## 12. Integration contract

Do not create a new dataset abstraction. Integrate the reservoir encoder into the existing repository and existing link-stream pipeline.

The coding agent may select function names, classes, and internal file boundaries consistent with the repository. It must preserve these behavioral contracts:

```text
fit
    consumes chronological fit transactions from the existing pipeline
    never consumes labels
    verifies reservoir contraction before processing
    builds reservoir-specific tree feature caches and forests

score stream
    consumes chronological transactions
    returns exactly one anomaly score per transaction
    uses pre-batch states
    commits state only after scoring

save/load
    persists frozen reservoir weights, forests, configuration, and contraction metadata
    re-verifies every reservoir after loading
```

Avoid repository-wide refactoring unrelated to replacing the TGN subsystem.

---

## 13. Configuration

Merge these settings into the repository's existing configuration system. Exact nesting may follow the existing codebase.

```yaml
seed: 42

reservoir:
  num_reservoirs: 8
  state_dim: 64
  transaction_embedding_dim: 64

  leak_rate: 0.20
  target_operator_norm: 0.95
  contraction_tolerance: 1.0e-6
  verify_operator_norm: true
  reject_invalid_loaded_weights: true

  recurrent_layout: orthogonal
  recurrent_density: 0.10
  recurrent_group_count: 4
  grouped_base_layout: random_dense_uniform

  input_scale: 0.50
  bias_scale: 0.10
  transaction_projection_scale: 1.0

  batch_size: 256
  unseen_node_state: zeros
  dtype: float32

time_encoding:
  dimension: 16
  min_frequency: 1.0e-4
  max_frequency: 1.0

motifs:
  enabled: [fan_out, fan_in, rapid_chain, cycle]
  context_decay_beta: 0.00001
  context_shrinkage_alpha: 5.0
  rapid_chain_window_seconds: 86400
  cycle_window_seconds: 604800
  lambda_by_motif:
    fan_out: 1.0
    fan_in: 1.0
    rapid_chain: 1.0
    cycle: 1.0

trees:
  trees_per_reservoir: 25
  subsample_size: 256
  max_depth: null
  max_split_retries: 8
  split_eps: 1.0e-8
```

Derived values that must be recorded:

```text
measured_operator_norm_r
analytical_kappa_r
contraction_verified_r
```

Recommended default bound:

\[
\eta=0.95,
\qquad
\gamma=0.20,
\qquad
\kappa\le0.99.
\]

The exact effective memory horizon must be treated as an empirical property; strict contraction guarantees fading-memory stability, not optimal anomaly-detection memory.

---

## 14. Numerical stability

Required safeguards:

1. preserve the existing robust preprocessing of transaction features;
2. use `log1p` for amounts and heavy-tailed count features where already appropriate;
3. clamp time deltas to non-negative values;
4. use bounded `tanh` reservoir states and transaction projections;
5. use non-affine LayerNorm on transaction embeddings;
6. divide motif-context L2 distance by `sqrt(d_h)`;
7. use epsilon in context denominators;
8. terminate tree nodes when response range is too small;
9. store tree thresholds as Python float64;
10. assert all scores and cached features are finite;
11. normalize recurrent matrices using the largest singular value;
12. verify the analytical contraction factor with tolerance;
13. reject NaN, infinite, zero-rank unintended, or unverifiable recurrent matrices;
14. perform spectral-norm verification in float64 when practical, even if runtime states use float32.

---

## 15. Complexity and scalability

Let:

- `N`: number of transactions;
- `R`: number of reservoir members;
- `d_s`: node state dimension;
- `d_h`: transaction embedding dimension;
- `K`: bounded motif-neighborhood size;
- `T`: trees per reservoir;
- `psi`: tree subsample size.

Approximate costs:

```text
stream representation:
    dense recurrent matrix: O(R * N * d_s^2)
    structured/sparse recurrent matrix: depends on non-zero count or layout

local motif computation:
    O(N * K)

forest construction:
    O(R * T * psi * log(psi))

one-transaction inference:
    reservoir update + embedding + O(R * T * log(psi)) + O(K)
```

Memory:

```text
node reservoir states:
    O(R * active_nodes * d_s)

bounded temporal adjacency/context state:
    O(active_nodes * K)

trees:
    O(R * T * psi)
```

For millions of transactions:

- use the existing chronological chunk pipeline;
- keep node states in sparse id-to-row storage or hash maps;
- update only active nodes;
- use bounded motif adjacency deques;
- persist tree features to memory-mapped arrays or Parquet when needed;
- use sparse, ring, orthogonal, or grouped layouts as controlled experiments;
- do not materialize a full dynamic adjacency matrix for reservoir updates.

---

## 16. Required unit tests

### 16.1 Causality tests

```text
- no transaction is committed before its complete mini-batch is scored
- all transactions in a mini-batch read the same pre-batch node states
- changing future transactions does not change earlier embeddings or scores
- sender and receiver order is preserved
```

### 16.2 Transaction-output tests

```text
- one input transaction produces one embedding per reservoir
- one input transaction produces one final anomaly score
- no node-level classification output is created
- new nodes receive deterministic zero states
```

### 16.3 Frozen-weight tests

```text
- every reservoir and transaction-projection parameter has requires_grad == False
- no optimizer exists in the project
- repeated runs with the same seed and layout are identical
- different seeds produce different transaction embeddings
```

### 16.4 Weight-layout tests

Create and validate every required layout:

```text
random_dense_uniform
random_dense_gaussian
orthogonal
symmetric
antisymmetric
ring
sparse_random
block_diagonal_grouped
```

Verify layout-specific structural properties before and after scaling where applicable.

### 16.5 Analytical contraction tests

For every reservoir member:

```text
- W_res has shape [d_s, d_s]
- measured operator norm <= configured target + tolerance
- configured target < 1
- kappa = (1 - gamma) + gamma * measured_operator_norm
- kappa < 1
```

For randomly generated endpoint states and fixed endpoint inputs, verify:

\[
\|F_a(s)-F_a(s')\|_2
\leq
\kappa\|s-s'\|_2+\text{tolerance}.
\]

### 16.6 Initial-state washout test

Run the same endpoint-input sequence from two distinct initial states and verify:

\[
\|s_n-s_n'\|_2
\leq
\kappa^n\|s_0-s_0'\|_2+\text{tolerance}.
\]

Use float64 for the verification path and a realistic numerical tolerance.

### 16.7 Batched node-state tests

```text
- an inactive node state is unchanged by unrelated transactions
- all events in a mini-batch read the same pre-batch state
- repeated-node candidates are aggregated deterministically by arithmetic mean
- a repeated-node star batch satisfies the same kappa contraction bound
- sender and receiver roles produce ordered endpoint inputs
- CPU and CUDA ESN bank outputs agree within float32 tolerance
```

### 16.8 Existing tree and motif tests

Keep all existing motif, tree, cold-start context, score-range, and synthetic anomaly tests. Replace references to TGN embeddings with reservoir transaction embeddings.

---

## 17. Integration tests

Create a synthetic stream with:

- mostly repeated normal pair behavior;
- occasional new-receiver fan-out bursts;
- rapid pass-through chains;
- short directed cycles;
- transaction labels for evaluation only.

Expected behavior:

```text
- suspicious motif events have shorter mean path lengths than repeated normal events
- suspicious motif events have higher anomaly scores
- removing motif increment lowers detection quality on motif anomalies
- replacing the proposed split with DIF coordinate split changes results
```

---

## 18. Experimental ablations required by the paper

The codebase must support the following switches while keeping the main method fixed.

### 18.1 Isolation contribution

```text
A0: IF on raw transaction features
A1: DIF-style random MLP embeddings + ordinary coordinate split
A2: contractive transaction graph reservoir embeddings + ordinary DIF coordinate split
A3: contractive transaction graph reservoir embeddings + context distance only
A4: contractive transaction graph reservoir embeddings + motif increment only
A5: full contractive reservoir + motif-conditioned split
```

The critical comparison is `A2` versus `A5`, because it isolates the contribution of the motif-conditioned tree split while keeping the same transaction reservoir representations.

### 18.2 Reservoir layout study

Run the same full detector with:

```text
W0: random_dense_uniform
W1: random_dense_gaussian
W2: orthogonal
W3: symmetric
W4: antisymmetric
W5: ring
W6: sparse_random
W7: block_diagonal_grouped
```

All layouts must use the same target operator norm and must pass the same contraction verification.

### 18.3 Memory/stability study

Vary:

```text
leak_rate
target_operator_norm
resulting analytical kappa
reservoir state dimension
number of reservoir members
```

Report the measured operator norm and analytical `kappa` for every run. Do not describe `kappa` as a learned quantity.

---

## 19. Implementation milestones

The existing data pipeline is already implemented and is not part of this work.

### Milestone 1: replace TGN with one contractive transaction reservoir

Deliver:

```text
- endpoint-wise sender/receiver state transition
- transaction embedding from pre-batch states
- zero-state new-node support
- strict mini-batch read/commit semantics
- frozen parameters
```

### Milestone 2: weight factory and theory verification

Deliver:

```text
- all required recurrent-matrix layouts
- operator-norm normalization
- analytical kappa computation
- initialization and load-time verification
- contraction and washout tests
```

### Milestone 3: reservoir ensemble integration

Deliver:

```text
- R independent reservoirs
- independent node-state stores
- output [R, B, d_h]
- compatibility with existing motif context computation
```

### Milestone 4: preserve tree pipeline

Deliver:

```text
- existing motif-conditioned split unchanged
- reservoir-specific tree feature cache
- forest fitting and online scoring
- A2-versus-A5 ablation
```

### Milestone 5: layout experiments and large-data runner

Deliver:

```text
- configuration-driven reservoir layouts
- contraction metadata in experiment outputs
- existing chunked fit and score commands
- serialization and deterministic reload
```

---

## 20. CLI contract

```bash
python -m src.cli.fit \
  --config configs/default.yaml \
  --input data/ibm_aml.parquet \
  --output artifacts/model
```

```bash
python -m src.cli.score \
  --model artifacts/model \
  --input data/ibm_aml_test.parquet \
  --output artifacts/scores.parquet
```

Output score file:

```text
event_id
timestamp
sender
receiver
anomaly_score
mean_path_length
label  # if available
```

---

## 21. Serialization

Persist the existing model artifacts plus reservoir-specific artifacts:

```text
model_config.yaml
feature_preprocessor artifact already used by the repository
reservoir_weights.pt
reservoir_contraction_report.json
forest.pkl
fit-end or configured runtime state
metadata.json
```

`reservoir_contraction_report.json` must contain one entry per reservoir member:

```json
{
  "reservoir_id": 0,
  "seed": 42,
  "layout": "orthogonal",
  "leak_rate": 0.2,
  "target_operator_norm": 0.95,
  "measured_operator_norm": 0.95,
  "analytical_kappa": 0.99,
  "verification_tolerance": 1e-6,
  "verification_passed": true
}
```

`metadata.json` must state:

```json
{
  "model_name": "TemporalMotifReservoirDIF",
  "version": "0.5.0",
  "detection_level": "transaction",
  "num_reservoirs": 8,
  "trees_per_reservoir": 25,
  "subsample_size": 256,
  "motifs": ["fan_out", "fan_in", "rapid_chain", "cycle"],
  "causal": true,
  "trained_neural_parameters": false,
  "strict_active_endpoint_contraction": true
}
```

On load:

1. restore weights;
2. recompute operator norm in the verification precision;
3. recompute `kappa`;
4. compare with stored metadata within tolerance;
5. reject the model if the strict bound fails.

---

## 22. Common implementation mistakes to avoid

```text
DO NOT modify or duplicate the existing dataset/link-stream pipeline.
DO NOT train the reservoir or transaction projection.
DO NOT implement TGN, GRU, LSTM, attention, or learned messages as the main encoder.
DO NOT use node anomaly classification as the target.
DO NOT output sender or receiver scores instead of a transaction score.
DO NOT update any state before the complete mini-batch has been scored.
DO NOT feed another node's reservoir state into an endpoint recurrence.
DO NOT treat unseen nodes as errors; initialize their states to zero.
DO NOT use spectral radius alone as the formal contraction guarantee.
DO NOT normalize recurrent sub-blocks independently and assume the full matrix is contractive.
DO NOT apply recurrent-weight perturbations after norm verification.
DO NOT claim ESP means infinite or lossless memory.
DO NOT shuffle the full dataset before temporal processing.
DO NOT use future edges in motif or context computation.
DO NOT replace the motif-conditioned split with an embedding-coordinate split.
DO NOT keep unlimited event history in RAM.
```

---

## 23. Minimal end-to-end pseudocode

The pseudocode below describes behavior, not mandatory class names.

```python
def initialize_reservoir_member(config, seed):
    W_raw = initialize_recurrent_layout(
        layout=config.recurrent_layout,
        size=config.state_dim,
        seed=seed,
        density=config.recurrent_density,
        group_count=config.recurrent_group_count,
    )

    sigma_max = largest_singular_value(W_raw, dtype="float64")
    W_res = config.target_operator_norm * W_raw / max(sigma_max, config.eps)

    measured_norm = largest_singular_value(W_res, dtype="float64")
    kappa = (
        1.0
        - config.leak_rate
        + config.leak_rate * measured_norm
    )

    assert measured_norm <= config.target_operator_norm + config.contraction_tolerance
    assert measured_norm < 1.0
    assert kappa < 1.0

    return frozen_reservoir_weights_and_report(...)
```

```python
def process_mini_batch(group, reservoir_member, node_state, temporal_state):
    # Read-only phase.
    src_old = node_state.read_or_zero(group.src)
    dst_old = node_state.read_or_zero(group.dst)

    transaction_input = build_transaction_input(
        edge_features=group.edge_features,
        src_time_gap=temporal_state.src_time_gap(group),
        dst_time_gap=temporal_state.dst_time_gap(group),
        src_is_new=node_state.is_unseen(group.src),
        dst_is_new=node_state.is_unseen(group.dst),
    )

    transaction_embedding = frozen_transaction_projection(
        concat(src_old, dst_old, transaction_input)
    )

    motif_increment = compute_all_motif_increments(group, temporal_state)
    context_distance = compute_motif_context_distances(
        group,
        transaction_embedding,
        temporal_state,
    )

    path_lengths_or_cache = use_existing_tree_pipeline(
        context_distance,
        motif_increment,
    )

    # Candidate transition from the same pre-batch states.
    endpoint_old = concat_rows(src_old, dst_old)              # [2B, d_s]
    endpoint_input = append_role_and_repeat(transaction_input) # [2B, d_a + 1]
    endpoint_candidate = tanh(
        endpoint_input @ W_in.T + endpoint_old @ W_res.T + bias
    )                                                         # [2B, d_s]
    endpoint_new = (
        (1 - leak_rate) * endpoint_old
        + leak_rate * endpoint_candidate
    )

    pending_node_updates = mean_by_node(
        concat(group.src, group.dst),
        endpoint_new,
    )

    return transaction_embedding, path_lengths_or_cache, pending_node_updates
```

```python
def commit_mini_batch(pending_updates, group, node_state, temporal_state, context_state):
    node_state.commit(pending_updates)
    context_state.commit(group)
    temporal_state.commit(group)
```

```python
def fit_model(existing_transaction_stream, config):
    reservoirs = [
        initialize_reservoir_member(config.reservoir, seed_for_member(r))
        for r in range(config.reservoir.num_reservoirs)
    ]

    feature_caches = []

    for reservoir in reservoirs:
        node_state = empty_sparse_node_state()
        temporal_state = empty_temporal_state()
        context_state = empty_context_state()
        cache = new_tree_feature_cache()

        for group in batch_existing_stream_chronologically(existing_transaction_stream):
            embedding, tree_features, pending = process_mini_batch(
                group,
                reservoir,
                node_state,
                temporal_state,
            )
            cache.append(tree_features)
            commit_mini_batch(
                pending,
                group,
                node_state,
                temporal_state,
                context_state,
            )

        feature_caches.append(cache.finalize())

    forest = fit_existing_motif_isolation_forest(feature_caches, config.trees)
    return assembled_model(reservoirs, forest, config)
```

---

## 24. Source alignment and theoretical positioning

This specification follows these source-level principles:

### Dynamic Graph Echo State Networks / Discrete-time DynGESN

- Dynamic graph echo-state reservoirs use randomly initialized, fixed input and recurrent weights.
- Their states are updated over graph time without training the recurrent reservoir.
- A leakage connection combines the previous state with a nonlinear candidate state.
- Echo State Property analysis is based on contraction or asymptotic contraction of the state transition.
- The original DynGESN formulation processes discrete-time graph snapshots; this specification adapts the same reservoir principle to asynchronous transaction events.

### 2024 graph echo-state reservoir design study

- Reservoir behavior depends on more than one scalar spectral-radius parameter.
- Dense random, orthogonal, symmetric, antisymmetric, ring, sparse, and grouped recurrent layouts produce different dynamics and representation richness.
- Orthogonal reservoirs are associated with improved memory capacity in sequence ESNs.
- Sparse and grouped layouts provide computational alternatives.

### DIF

- DIF motivates an ensemble of randomly initialized, optimization-free nonlinear representation spaces.
- In this method, the random representation generator is a contractive event-driven transaction graph reservoir rather than a frozen TGN or random MLP.
- No supervised reservoir readout is trained; the downstream model is the motif-conditioned Isolation Forest.

### Proposed contribution beyond the sources

The source papers do not directly provide a transaction-event anomaly detector for sender-receiver link streams. The proposed method contributes:

1. a transaction-level event-driven adaptation of dynamic graph echo-state reservoirs;
2. strict operator-norm normalization that yields a transparent active-node mini-batch contraction bound;
3. asynchronous node-wise fading-memory interpretation for transaction streams;
4. integration with the temporal motif-conditioned isolation split defined in Section 9.

The theoretical guarantee is a sufficient strict contraction guarantee for every active-node mini-batch update. It must not be presented as a guarantee of infinite memory, optimal memory capacity, or anomaly-detection accuracy.

---

## 25. Definition of done

The implementation is complete when all conditions below hold.

```text
[ ] The existing dataset and link-stream pipeline remains unchanged.
[ ] Every transaction receives one embedding per reservoir member.
[ ] Every transaction receives one final anomaly score.
[ ] Sender and receiver states are used only as historical context.
[ ] Unseen nodes are supported with deterministic zero states.
[ ] No TGN, GRU/LSTM, attention, optimizer, loss, or backpropagation is used in the main encoder.
[ ] All reservoir and transaction-projection weights are frozen.
[ ] Every recurrent matrix layout can be initialized from configuration.
[ ] Every recurrent matrix is normalized by its operator 2-norm.
[ ] Every reservoir member satisfies measured_operator_norm < 1.
[ ] Every reservoir member satisfies analytical_kappa < 1.
[ ] Contraction verification runs after initialization and model loading.
[ ] Every transaction in a mini-batch uses the same pre-batch state.
[ ] Inactive node states are unchanged by unrelated transactions.
[ ] Four motif context distances and four motif increments are produced per transaction.
[ ] Every tree node uses the unchanged motif-conditioned response.
[ ] The forest outputs finite transaction anomaly scores in [0, 1].
[ ] The model can be saved, loaded, re-verified, and reproduce identical scores.
[ ] Unit and integration tests pass.
[ ] A2-versus-A5 ablation uses identical reservoir transaction embeddings.
[ ] Reservoir-layout experiments use the same contraction target and verification protocol.
```

---
