# Contextual Temporal Deep Isolation Forest for Link Streams

## 0. Goal

We want an optimization-free anomaly detector for temporal link streams.

Each sample is a single temporal link event:

\[
e_i = (t_i, u_i, v_i, x_i)
\]

where:

- \(t_i\): timestamp
- \(u_i\): source node / sender
- \(v_i\): destination node / receiver
- \(x_i \in \mathbb{R}^{d_e}\): edge feature vector

The method follows the high-level DIF idea:

\[
\text{link stream batch}
\rightarrow
\text{random neural encoders}
\rightarrow
\text{random representations}
\rightarrow
\text{isolation forest}
\]

But we replace the ordinary DIF split:

\[
h_{i,k} < \theta
\]

with a new **Contextual Temporal Isolation Split**:

\[
a^\top \Gamma_i < \theta
\]

where \(\Gamma_i\) compares the current event representation against sender history, receiver history, and pair history.

---

## 1. Notation

### 1.1 Stream

The full link stream is:

\[
\mathcal{L} = \{e_i\}_{i=1}^{N}
\]

with events sorted by time:

\[
t_1 \leq t_2 \leq \dots \leq t_N
\]

Each event:

\[
e_i = (t_i, u_i, v_i, x_i)
\]

where:

\[
x_i \in \mathbb{R}^{d_e}
\]

### 1.2 Mini-batch

A mini-batch is:

\[
B = \{e_i\}_{i=b_0}^{b_0+B_s-1}
\]

Use \(B_s\) for batch size.

Tensor representation:

```python
src:        LongTensor[B_s]        # source node ids
dst:        LongTensor[B_s]        # destination node ids
ts:         FloatTensor[B_s]       # timestamps
edge_feat:  FloatTensor[B_s, d_e]  # edge features
```

---

## 2. Random Temporal Encoder

We use \(R\) random encoders:

\[
F^{(1)}, F^{(2)}, \dots, F^{(R)}
\]

Each encoder is initialized randomly and never trained.

Two possible encoder types:

1. Random MLP encoder
2. Random TGN-style memory encoder

The TGN-style version is more aligned with temporal graphs.

---

## 3. Random TGN-style Encoder

### 3.1 Node memory

For each random encoder \(r\), maintain node memory:

\[
s_u^{(r)}(t) \in \mathbb{R}^{d_s}
\]

Memory store:

```python
memory[r]: Dict[node_id, FloatTensor[d_s]]
```

For unseen node \(u\):

\[
s_u^{(r)}(t^-) = \mathbf{0} \in \mathbb{R}^{d_s}
\]

For one batch:

```python
s_src = memory_lookup(src)  # FloatTensor[B_s, d_s]
s_dst = memory_lookup(dst)  # FloatTensor[B_s, d_s]
```

Shapes:

\[
S_{\text{src}}^{(r)} \in \mathbb{R}^{B_s \times d_s}
\]

\[
S_{\text{dst}}^{(r)} \in \mathbb{R}^{B_s \times d_s}
\]

---

### 3.2 Temporal features

For each event \(e_i=(t_i,u_i,v_i,x_i)\), compute basic temporal features:

\[
\Delta u_i = t_i - t_{\text{last}}(u_i)
\]

\[
\Delta v_i = t_i - t_{\text{last}}(v_i)
\]

\[
\Delta uv_i = t_i - t_{\text{last}}(u_i,v_i)
\]

Also optionally include window counts:

\[
c_u^W(t_i) = \#\{(t,u_i,\cdot): t \in [t_i-W,t_i)\}
\]

\[
c_v^W(t_i) = \#\{(t,\cdot,v_i): t \in [t_i-W,t_i)\}
\]

\[
c_{uv}^W(t_i) = \#\{(t,u_i,v_i): t \in [t_i-W,t_i)\}
\]

Let the temporal-statistical feature vector be:

\[
\tau_i \in \mathbb{R}^{d_\tau}
\]

Example:

\[
\tau_i =
[
\log(1+\Delta u_i),
\log(1+\Delta v_i),
\log(1+\Delta uv_i),
\log(1+c_u^W),
\log(1+c_v^W),
\log(1+c_{uv}^W)
]
\]

Then:

\[
d_\tau = 6
\]

Batch tensor:

```python
time_feat: FloatTensor[B_s, d_tau]
```

---

### 3.3 Event input vector

For encoder \(r\), define event input:

\[
q_i^{(r)}
=
[
s_{u_i}^{(r)}(t_i^-)
\|
s_{v_i}^{(r)}(t_i^-)
\|
x_i
\|
\tau_i
]
\]

Shape:

\[
q_i^{(r)} \in \mathbb{R}^{d_q}
\]

where:

\[
d_q = 2d_s + d_e + d_\tau
\]

Batch tensor:

\[
Q^{(r)} \in \mathbb{R}^{B_s \times d_q}
\]

```python
q = concat([s_src, s_dst, edge_feat, time_feat], dim=-1)
# q.shape == [B_s, d_q]
```

---

### 3.4 Random representation

Each random encoder maps:

\[
F^{(r)}: \mathbb{R}^{d_q} \rightarrow \mathbb{R}^{d_h}
\]

\[
h_i^{(r)} = F^{(r)}(q_i^{(r)})
\]

Batch:

\[
H^{(r)} \in \mathbb{R}^{B_s \times d_h}
\]

```python
h = random_encoder[r](q)
# h.shape == [B_s, d_h]
```

No backpropagation. No gradient update.

---

### 3.5 Memory update

After computing \(h_i^{(r)}\), update node memories.

Define random message function:

\[
m_i^{(r)} =
\phi_m^{(r)}(q_i^{(r)})
\]

\[
m_i^{(r)} \in \mathbb{R}^{d_m}
\]

Batch:

\[
M^{(r)} \in \mathbb{R}^{B_s \times d_m}
\]

For source node:

\[
m_{u_i}^{(r)} = m_i^{(r)}
\]

For destination node, either use the same message or another random function:

\[
m_{v_i}^{(r)} =
\phi_m^{\prime(r)}(q_i^{(r)})
\]

If a node appears multiple times in the batch, aggregate messages:

\[
\bar{m}_u^{(r)} =
\text{mean}\{m_i^{(r)}: u_i=u \text{ or } v_i=u\}
\]

Update memory:

\[
s_u^{(r)}(t_i)
=
\phi_{\text{upd}}^{(r)}
\left(
s_u^{(r)}(t_i^-),
\bar{m}_u^{(r)}
\right)
\]

Example update:

\[
s_u^{(r)}(t_i)
=
\tanh
\left(
W_s^{(r)}s_u^{(r)}(t_i^-)
+
W_m^{(r)}\bar{m}_u^{(r)}
+
b^{(r)}
\right)
\]

All weights are random and frozen.

---

## 4. Historical Context Statistics

For each random encoder \(r\), we maintain historical statistics in the representation space \(h^{(r)}\).

For each sender \(u\):

\[
\mu_u^{(r)} \in \mathbb{R}^{d_h},
\quad
\sigma_u^{(r)} \in \mathbb{R}^{d_h}
\]

For each receiver \(v\):

\[
\mu_v^{(r)} \in \mathbb{R}^{d_h},
\quad
\sigma_v^{(r)} \in \mathbb{R}^{d_h}
\]

For each directed pair \((u,v)\):

\[
\mu_{uv}^{(r)} \in \mathbb{R}^{d_h},
\quad
\sigma_{uv}^{(r)} \in \mathbb{R}^{d_h}
\]

These statistics are computed only from past events.

For event \(e_i\), use:

\[
\mu_{u_i}^{(r)}(t_i^-), \sigma_{u_i}^{(r)}(t_i^-)
\]

\[
\mu_{v_i}^{(r)}(t_i^-), \sigma_{v_i}^{(r)}(t_i^-)
\]

\[
\mu_{u_iv_i}^{(r)}(t_i^-), \sigma_{u_iv_i}^{(r)}(t_i^-)
\]

For unseen sender / receiver / pair, use default:

\[
\mu = \mathbf{0}, \quad \sigma = \mathbf{1}
\]

---

## 5. Contextual Representation

Given random representation:

\[
h_i^{(r)} \in \mathbb{R}^{d_h}
\]

Compute sender-normalized deviation:

\[
z_{u,i}^{(r)}
=
\frac{
h_i^{(r)} - \mu_{u_i}^{(r)}
}{
\sigma_{u_i}^{(r)} + \epsilon
}
\]

Receiver-normalized deviation:

\[
z_{v,i}^{(r)}
=
\frac{
h_i^{(r)} - \mu_{v_i}^{(r)}
}{
\sigma_{v_i}^{(r)} + \epsilon
}
\]

Pair-normalized deviation:

\[
z_{uv,i}^{(r)}
=
\frac{
h_i^{(r)} - \mu_{u_iv_i}^{(r)}
}{
\sigma_{u_iv_i}^{(r)} + \epsilon
}
\]

Also compute raw differences:

\[
d_{u,i}^{(r)} = h_i^{(r)} - \mu_{u_i}^{(r)}
\]

\[
d_{v,i}^{(r)} = h_i^{(r)} - \mu_{v_i}^{(r)}
\]

\[
d_{uv,i}^{(r)} = h_i^{(r)} - \mu_{u_iv_i}^{(r)}
\]

Define contextual representation:

\[
\Gamma_i^{(r)}
=
[
h_i^{(r)}
\|
z_{u,i}^{(r)}
\|
z_{v,i}^{(r)}
\|
z_{uv,i}^{(r)}
\|
d_{u,i}^{(r)}
\|
d_{v,i}^{(r)}
\|
d_{uv,i}^{(r)}
]
\]

Shape:

\[
\Gamma_i^{(r)} \in \mathbb{R}^{d_c}
\]

where:

\[
d_c = 7d_h
\]

Batch:

\[
\Gamma^{(r)} \in \mathbb{R}^{B_s \times d_c}
\]

```python
gamma = concat([
    h,
    (h - mu_src) / (std_src + eps),
    (h - mu_dst) / (std_dst + eps),
    (h - mu_pair) / (std_pair + eps),
    h - mu_src,
    h - mu_dst,
    h - mu_pair,
], dim=-1)

# gamma.shape == [B_s, 7 * d_h]
```

---

## 6. New Tree Split: Contextual Temporal Isolation Split

Ordinary DIF split:

\[
h_{i,k}^{(r)} < \theta
\]

Proposed split:

\[
a^\top \Gamma_i^{(r)} < \theta
\]

where:

\[
a \in \mathbb{R}^{d_c}
\]

is a random projection vector.

At each tree node:

1. Current sample index set:

\[
A \subseteq \{1,\dots,N\}
\]

2. Current contextual representations:

\[
\Gamma_A^{(r)} \in \mathbb{R}^{|A| \times d_c}
\]

3. Sample random projection:

\[
a \sim \mathcal{N}(0, I_{d_c})
\]

Optionally use sparse random projection for scalability:

\[
a_j =
\begin{cases}
+1 & \text{with probability } p/2 \\
-1 & \text{with probability } p/2 \\
0 & \text{with probability } 1-p
\end{cases}
\]

4. Project all samples:

\[
y_i = a^\top \Gamma_i^{(r)}
\]

Shape:

\[
y_A \in \mathbb{R}^{|A|}
\]

5. Choose random threshold:

\[
\theta \sim \text{Uniform}
\left(
\min_{i\in A} y_i,
\max_{i\in A} y_i
\right)
\]

6. Split:

\[
A_L = \{i \in A: a^\top \Gamma_i^{(r)} < \theta\}
\]

\[
A_R = \{i \in A: a^\top \Gamma_i^{(r)} \geq \theta\}
\]

This is the main innovation.

The split is not performed in raw feature space or ordinary random representation space. It is performed in a context-normalized temporal link space.

---

## 7. Forest Construction

### 7.1 Hyperparameters

```python
R: int          # number of random encoders
T: int          # number of trees per encoder
psi: int        # subsample size per tree
max_depth: int  # usually ceil(log2(psi))
d_s: int        # node memory dimension
d_e: int        # edge feature dimension
d_tau: int      # temporal feature dimension
d_q: int        # input dimension to random encoder
d_h: int        # random representation dimension
d_c: int        # contextual representation dimension
```

Typical values:

```python
R = 8
T = 25
psi = 256
max_depth = ceil(log2(psi))
d_s = 32
d_tau = 6
d_h = 64
d_c = 7 * d_h = 448
```

If:

\[
d_e = 10
\]

then:

\[
d_q = 2d_s + d_e + d_\tau
\]

\[
d_q = 2 \times 32 + 10 + 6 = 80
\]

So the random encoder maps:

\[
F^{(r)}: \mathbb{R}^{80} \rightarrow \mathbb{R}^{64}
\]

and the split operates on:

\[
\Gamma_i^{(r)} \in \mathbb{R}^{448}
\]

---

### 7.2 Offline fitting

Input:

```python
events:
    src:       LongTensor[N]
    dst:       LongTensor[N]
    ts:        FloatTensor[N]
    edge_feat: FloatTensor[N, d_e]
```

Output:

```python
forest: List[ContextualIsolationTree]
```

Algorithm:

```python
for r in range(R):

    initialize random encoder F_r
    initialize node memory store M_r
    initialize context statistics:
        sender_stats_r
        receiver_stats_r
        pair_stats_r

    Gamma_r = []

    for batch in stream_in_time_order(events):

        # 1. Read batch
        src, dst, ts, edge_feat = batch

        # 2. Lookup old node memories
        s_src = M_r.lookup(src)  # [B_s, d_s]
        s_dst = M_r.lookup(dst)  # [B_s, d_s]

        # 3. Compute temporal features
        time_feat = compute_temporal_features(src, dst, ts)  # [B_s, d_tau]

        # 4. Build random encoder input
        q = concat([s_src, s_dst, edge_feat, time_feat], dim=-1)
        # [B_s, d_q]

        # 5. Random representation
        h = F_r(q)
        # [B_s, d_h]

        # 6. Lookup historical context statistics
        mu_src, std_src = sender_stats_r.lookup(src)  # [B_s, d_h]
        mu_dst, std_dst = receiver_stats_r.lookup(dst)  # [B_s, d_h]
        mu_pair, std_pair = pair_stats_r.lookup(src, dst)  # [B_s, d_h]

        # 7. Build contextual representation
        gamma = concat([
            h,
            (h - mu_src) / (std_src + eps),
            (h - mu_dst) / (std_dst + eps),
            (h - mu_pair) / (std_pair + eps),
            h - mu_src,
            h - mu_dst,
            h - mu_pair,
        ], dim=-1)
        # [B_s, 7 * d_h]

        Gamma_r.append(gamma)

        # 8. Update memory AFTER representation extraction
        msg = random_message_r(q)  # [B_s, d_m]
        M_r.update(src, dst, msg)

        # 9. Update historical statistics AFTER gamma extraction
        sender_stats_r.update(src, h)
        receiver_stats_r.update(dst, h)
        pair_stats_r.update(src, dst, h)

    Gamma_r = concat(Gamma_r, dim=0)
    # [N, d_c]

    # 10. Build T trees for this encoder
    for t in range(T):
        idx = random_subsample(N, size=psi)
        tree = build_contextual_itree(Gamma_r[idx], idx, max_depth)
        forest.append((r, tree))
```

Important rule:

```text
For each event, compute h and gamma using only past memory and past statistics.
Only after that, update memory and statistics with the current event.
```

This prevents temporal leakage.

---

## 8. Tree Building Procedure

Input:

```python
Gamma: FloatTensor[psi, d_c]
indices: LongTensor[psi]
max_depth: int
```

Output:

```python
tree: ContextualIsolationTree
```

Recursive algorithm:

```python
def build_node(Gamma_A, indices_A, depth):

    if depth >= max_depth:
        return Leaf(size=len(indices_A))

    if len(indices_A) <= 1:
        return Leaf(size=len(indices_A))

    # Random contextual projection
    a = sample_random_projection(d_c)  # [d_c]

    y = Gamma_A @ a  # [len(indices_A)]

    y_min = y.min()
    y_max = y.max()

    if y_min == y_max:
        return Leaf(size=len(indices_A))

    theta = uniform(y_min, y_max)

    left_mask = y < theta
    right_mask = ~left_mask

    if left_mask.sum() == 0 or right_mask.sum() == 0:
        return Leaf(size=len(indices_A))

    left = build_node(Gamma_A[left_mask], indices_A[left_mask], depth + 1)
    right = build_node(Gamma_A[right_mask], indices_A[right_mask], depth + 1)

    return InternalNode(
        projection=a,
        threshold=theta,
        left=left,
        right=right
    )
```

Each internal node stores:

```python
projection: FloatTensor[d_c]
threshold: float
left: Node
right: Node
```

---

## 9. Inference

For a new event:

\[
e_* = (t_*, u_*, v_*, x_*)
\]

For each random encoder \(r\):

1. Lookup memory:

\[
s_{u_*}^{(r)}(t_*^-), s_{v_*}^{(r)}(t_*^-)
\]

2. Build:

\[
q_*^{(r)} \in \mathbb{R}^{d_q}
\]

3. Compute random representation:

\[
h_*^{(r)} = F^{(r)}(q_*^{(r)}) \in \mathbb{R}^{d_h}
\]

4. Lookup historical context statistics:

\[
\mu_{u_*}^{(r)}, \sigma_{u_*}^{(r)}
\]

\[
\mu_{v_*}^{(r)}, \sigma_{v_*}^{(r)}
\]

\[
\mu_{u_*v_*}^{(r)}, \sigma_{u_*v_*}^{(r)}
\]

5. Build contextual representation:

\[
\Gamma_*^{(r)} \in \mathbb{R}^{d_c}
\]

6. Traverse all trees associated with encoder \(r\).

At each tree node:

\[
y_* = a^\top \Gamma_*^{(r)}
\]

If:

\[
y_* < \theta
\]

go left; otherwise go right.

The path length in tree \(j\) is:

\[
\ell_j(e_*)
\]

Average path length:

\[
\bar{\ell}(e_*) =
\frac{1}{RT}
\sum_{r=1}^{R}
\sum_{j=1}^{T}
\ell_{r,j}(e_*)
\]

Anomaly score:

\[
S(e_*) =
2^{-\frac{\bar{\ell}(e_*)}{c(\psi)}}
\]

where \(c(\psi)\) is the standard isolation forest normalization constant.

Higher score means more anomalous.

After scoring, update memory and context statistics if using online deployment.

---

## 10. Key Innovation

### Ordinary iForest

\[
\text{split}(x_i)
=
\mathbf{1}[x_{i,k}<\theta]
\]

### DIF

\[
\text{split}(x_i)
=
\mathbf{1}[F_r(x_i)_k<\theta]
\]

### Proposed method

\[
\text{split}(e_i)
=
\mathbf{1}
[
a^\top
\Gamma_i^{(r)}
<
\theta
]
\]

where:

\[
\Gamma_i^{(r)}
=
\Gamma
\left(
F^{(r)}
(
\psi(e_i,\mathcal{L}_{<t_i})
),
\text{sender history},
\text{receiver history},
\text{pair history}
\right)
\]

In words:

```text
DIF isolates points in random representation space.
This method isolates temporal links in random context-deviation space.
```

The split is contextual because the same transaction representation \(h_i\) can be normal for one sender but abnormal for another sender.

---

## 11. Implementation Classes

Recommended code structure:

```python
class RandomTGNEncoder:
    def __init__(self, d_s, d_e, d_tau, d_h, d_m):
        ...
    def encode_batch(self, src, dst, ts, edge_feat):
        # returns h, q, time_feat
        ...
    def update_memory(self, src, dst, q):
        ...
```

```python
class RunningContextStats:
    def lookup_sender(self, src):
        # returns mu, std
        ...
    def lookup_receiver(self, dst):
        ...
    def lookup_pair(self, src, dst):
        ...
    def update_sender(self, src, h):
        ...
    def update_receiver(self, dst, h):
        ...
    def update_pair(self, src, dst, h):
        ...
```

```python
class ContextualIsolationTree:
    def fit(self, Gamma):
        ...
    def path_length(self, gamma):
        ...
```

```python
class ContextualTemporalDIF:
    def fit(self, events):
        ...
    def score_batch(self, batch):
        ...
    def update_after_scoring(self, batch):
        ...
```

---

## 12. Minimal Tensor Example

Assume:

```python
B_s = 1024
d_e = 10
d_s = 32
d_tau = 6
d_h = 64
```

Then:

```python
src.shape        == [1024]
dst.shape        == [1024]
edge_feat.shape  == [1024, 10]
s_src.shape      == [1024, 32]
s_dst.shape      == [1024, 32]
time_feat.shape  == [1024, 6]
q.shape          == [1024, 80]
h.shape          == [1024, 64]
mu_src.shape     == [1024, 64]
std_src.shape    == [1024, 64]
mu_dst.shape     == [1024, 64]
std_dst.shape    == [1024, 64]
mu_pair.shape    == [1024, 64]
std_pair.shape   == [1024, 64]
gamma.shape      == [1024, 448]
```

Tree split:

```python
a.shape          == [448]
y = gamma @ a
y.shape          == [1024]
theta            == scalar
left_mask.shape  == [1024]
right_mask.shape == [1024]
```

---

## 13. First Prototype Scope

For the first prototype, implement only:

1. Single-link event samples
2. Random TGN-style encoder
3. Sender / receiver / pair running statistics
4. Contextual representation \(\Gamma\)
5. Oblique contextual split \(a^\top \Gamma_i < \theta\)
6. Standard isolation path-length scoring

Do not implement yet:

- graph attention
- k-hop temporal neighbor retrieval
- backpropagation
- trainable TGN
- complex motif features
- KNN density split

The first prototype should stay optimization-free and stream-friendly.
