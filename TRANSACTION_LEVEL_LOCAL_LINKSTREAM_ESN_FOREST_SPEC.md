# Transaction-Level Local Link-Stream ESN Forest Specification

## Status and scope

This document fixes the data, timestamp semantics, ESN encoding, and tree-population contract for the proposed **transaction-level local link-stream ESN forest**. It specifies how a target transaction obtains its historical local link stream, how equal-time events are handled atomically, how that stream is encoded into an ESN state trajectory, what one tree receives as its training population, and the three label-free ESN split families used by the forest. Handcrafted historical features and SHAP-selected features are excluded from this model and reserved for separate IF baselines.

The prediction unit is always one directed transaction pair, never an account or graph. A seed is an edge with two endpoints, not two independent node queries.

## 1. Chronological data split

Store IBM AML transactions in deterministic `(timestamp, original_row_id)` order and split chronologically:

```text
first 70%  -> train
last  30%  -> test
```

The secondary `original_row_id` key is only a reproducible storage tie-breaker; it never creates causal precedence within an equal-time bucket. The training split supplies seed transactions used to construct local trees. Test transactions are held out for out-of-sample evaluation. Labels are never used in sampling.

## 2. Seed batch

Sample `batch_size = 64` seed transactions from the training split. The 64 seeds form the training population of **one tree**. For seed `i`:

```text
e_i = (u_i -> v_i @ t_i)
```

The batch is a set of independent sampling requests, not a chronological stream update:

```text
batch = {(e_i, L_i)} for i = 1..64
```

Each seed has its own local link stream `L_i`; its size `N_i = |L_i|` may differ from every other seed, including `N_i = 0`. A seed is never removed merely because it has little or no historical context.

## 3. Strictly-past seeded temporal k-hop neighborhood

For seed `e_i = (u_i -> v_i @ t_i)`, start from the endpoint frontier:

```text
frontier_0 = {u_i, v_i}
```

Expand for a configured `k` hops. At each hop, for every frontier account, retrieve historical incident transactions (both incoming and outgoing) satisfying:

```text
t_i - 7 days <= edge.timestamp < t_i
```

The endpoints of selected historical edges become the next frontier. Deduplicate edges within this seed's hop expansion. Contexts for different seeds may overlap but remain independent. The target transaction itself, every future transaction, and **every transaction with `edge.timestamp == t_i`** are excluded.

Example:

```text
seed: A -> B @ t

hop 1: D -> A, A -> X, Y -> B, B -> C
hop 2: P -> D, X -> Q, C -> E, ...
```

### Sampling direction versus modelling direction

Sampling is **direction-agnostic incident expansion**: both `X -> A` and `A -> X` are valid historical context for endpoint `A`; likewise for `B` and later-hop accounts. This finds incoming and outgoing money-flow context without missing either side of an account's activity.

The selected edge is never made undirected in the returned link stream. It retains its original ordered fields:

```text
src, dst, timestamp, amount, currencies, payment format, edge attributes
```

Thus `A -> B` and `B -> A` remain distinct transactions during ESN encoding and later modelling.

```text
sampling: incident / direction-agnostic, to find complete context
modelling: directed, to preserve money-flow semantics
```


Therefore `L_i` is not the direct pair history of `A -> B`; it contains other transaction pairs that are temporally and topologically connected to the seed's endpoints.

## 4. Context-size policy

After k-hop expansion:

1. sort selected edges by `(timestamp, event_id)`; `event_id` is deterministic presentation only inside an equal-time bucket;
2. retain at most `max_context_events = 500` edges, keeping the 500 closest to the seed time;
3. retain the seed even when no historical edge remains.

```text
0 <= N_i <= 500    -> one valid seed request
N_i = 0            -> empty historical context, not an anomaly label
```

`context_size` and an `empty-context` flag may be recorded only for coverage and limitation analysis. They do not route a sample, enter a tree split, or create an anomaly score. Record their rate by split and label only during evaluation, to detect selection bias.

## 5. Batch representation

The contexts are ragged, not a dense tensor:

```text
seed_events:       [64, ...]
context_edges:     [sum(N_i), ...]
context_batch_id:  [sum(N_i)]   # values 0..63
offsets:           [65]
context_size:      [64]
valid:             [64]
```

For seed `i`, its historical link-stream edges occupy:

```text
context_edges[offsets[i] : offsets[i + 1]]
```

This represents `(batch_size, N_linkstream_i)` without padding.

## 6. Relation to TGN batching

The reference TGN implementation is **node-centric**: it queries a single node at a cutoff time and samples a fixed number `K` of temporal neighbors per node. Different true neighbor counts are padded to `[batch_size, K]` and ignored by an attention mask. A multi-hop TGN repeats this fixed-fanout query recursively.

This detector is **transaction-pair-centric**:

```text
TGN query:       context(node, time)
this query:      context(src, dst, time)
```

The pair seed starts with both endpoints `{src, dst}`. Its selected historical transaction edges are encoded as one local temporal context; they are not themselves the tree's training population. The resulting tree still receives exactly 64 seed-transaction representations.

Use the existing ragged representation instead:

```text
flat context_edges + context_batch_id + offsets
```

It allows a batch of 64 independent seeds to be processed without padding, while still permitting parallel work. `offsets[i]:offsets[i+1]` identifies the complete link stream for seed `i`; an ESN must reset its per-seed temporary state at each segment boundary. Padding to `[64, N_max, d]` with a mask is an equivalent execution representation when needed for vectorized ESN evaluation. Optional length bucketing is an execution optimization only and must not change which edges are sampled.


## 7. Required invariants
- Every context edge has `edge.timestamp < seed.timestamp`; equal-time edges are never historical context.
- No label participates in seed selection or neighborhood extraction.
- All contexts use the same seven-day window, hop count, direction policy, and 500-edge cap for a declared experiment.
- Context timestamps remain `float64`; IBM epoch timestamps must not be downcast to `float32`.
- Contexts belonging to different seeds are independent, even if they contain overlapping historical edges; any later ESN state is reset at the context boundary.
- The seed transaction is not a member of its own historical context. It is later encoded as the transaction to be scored, conditioned only on that past context.
- Sampling treats edges as incident; modelling always retains the original `src -> dst` direction.

## 8. TGN versus IBM AML data semantics

TGN commonly supports two input sources:

```text
node features: static or learned attributes for each node
edge features: attributes of one interaction
```

IBM AML naturally provides rich **transaction-edge features**, but no account-profile table. Its accounts are nodes identified by `(bank_id, account_id)`; raw transaction fields are timestamp, paid/received amount, payment/receiving currency, payment format, and the two endpoint banks. Account IDs are identifiers, not ordinal numeric node features.

```text
TGN-style data:   node features + edge interaction features
IBM AML raw data: node identifiers + directed transaction-edge features
```

The initial local model therefore uses directed raw transaction features as ESN input. It does not fabricate static account profiles or feed account IDs as arbitrary numeric features. Handcrafted historical account statistics remain separate IF-baseline features, not inputs or split rules of this ESN forest.

## 9. Per-account memory decision and concern

TGN's optional node memory is useful when nodes have stable semantics and the model is trained to maintain a long-lived node representation. IBM AML lacks raw account attributes, while this detector's target is a transaction pair relative to its sampled local link stream.

The concern is that persistent `memory[account]` would turn the design toward a global account-behaviour model: it is more complex, expensive at IBM scale, and may make account identity/history dominate the transaction-level objective.

The first local ESN variant therefore has:

```text
one temporary shared ESN state per seed context L_i
no global per-account ESN memory
no per-seed, per-account neural memory
```

The state is reset before each `L_i`, processes its directed historical edges in chronological order, and is discarded after that context is encoded. This retains a local money-flow sequence representation while keeping the prediction and tree population transaction-level.

This is a testable simplification, not a claim that account memory can never help. A later ablation may compare the same sampler, raw inputs, tree design, and score with a temporary local per-account state; add it only if the stateless local ESN baseline shows value.

## 10. Local ESN to tree contract

For one tree, sample 64 training seeds and build one strictly-past local link stream for each seed. Each context is processed independently in chronological **timestamp-bucket** order by the same ESN:

```text
L_i = [historical edge_1, ..., historical edge_Ni]
z_i^(0) = 0
L_i -> timestamp-bucket ESN -> c_i
seed transaction x_i + c_i -> seed encoder -> h_i
```

`c_i` is the local historical-context embedding. `h_i` is the representation of the target seed transaction conditioned on its past context. If `N_i = 0`, use a fixed empty state (for example `c_empty = 0`) and still encode the seed transaction; no future edge or target edge may be used to fill the empty context. Section 11.2a defines the atomic handling of equal-time edges.

There is one temporary ESN state per seed context, reset before `L_i` and discarded after that context is encoded. There is no persistent global account memory and no persistent per-account neural state in the first variant.

The forest's trees receive the following ESN outputs for each of the 64 seed-level samples:

```text
{(Z_i, c_i, h_i) | i = 1..64}, where Z_i = [C_i1, ..., C_iRi]
```

The base target transition is `d_i = h_i-c_i`. The `N_i` historical edges in `L_i` are never treated as the individual training samples for a tree; they exist only to construct `Z_i`. The next tree independently resamples 64 training seeds; seed reuse across trees is allowed. Section 11 specifies how the three split families read these ESN outputs.

## 11. ESN-only split and scoring pipeline

### 11.1 ESN input

Each historical transaction edge is represented by its own raw transaction fields with safe numerical or categorical preparation, for example:

```text
log1p(amount_paid), log1p(amount_received), same_bank, same_currency,
payment-format encoding, currency encoding, cyclic time encoding, direction
```

These transformations make raw fields usable by an ESN; they are not AML-specific hand-engineered IF features. Do not feed `is_new_pair`, historical counts, historical amount ratios, time-gap features, SHAP-selected features, or labels into the ESN or any tree split.

### 11.2 Per-seed ESN state trajectory

For a seed `i`, process its strictly-past context in chronological timestamp-bucket order with a fresh local state:

```text
C_i^(0) = 0
same-time bucket B_i1 -> C_i1
same-time bucket B_i2 -> C_i2
...
same-time bucket B_iRi -> C_iRi = c_i
seed transaction x_i -> h_i = ESN_step(x_i, c_i)
```

`Z_i = [C_i1, ..., C_iRi]` is the complete historical ESN state trajectory, with one state per distinct historical timestamp and `R_i <= N_i`. `c_i` summarizes the final historical context. `h_i` is the target transaction state conditioned on that past context. The target edge is encoded after the historical stream; it is never inserted into `L_i`.

When `L_i` is empty:

```text
c_i = c_empty (initially the all-zero ESN state)
h_i = ESN_step(x(seed_i), c_empty)
d_i = h_i
```

The initial research question deliberately focuses on contexts with sufficient history for trajectory-based splits. Sparse or empty contexts are a reported coverage limitation, not a hand-feature routing rule in this specification.

### 11.2a Atomic equal-timestamp ESN update

IBM AML timestamps are minute-granular, so equal timestamps are treated as simultaneous rather than as a hidden CSV ordering. Partition one historical local stream into timestamp buckets:

```text
B_i,r = {edge in L_i : edge.timestamp = tau_i,r}
tau_i,1 < ... < tau_i,Ri < seed.timestamp
```

All edges in one bucket read the same preceding context state `C_(tau_i,r-)`. Compute one candidate per directed edge, mean-pool those candidates, then commit once:

\[
u_{i,r,e}=\tanh(W_{in}x_{i,r,e}+W_{res}C_{i,r^-}+b),
\qquad
C_{i,r}=(1-\alpha)C_{i,r^-}+\alpha\,\operatorname{mean}_{e\in B_{i,r}}u_{i,r,e}.
\]

The mean is permutation-invariant: reordering equal-time CSV rows cannot change `C_i`. The stored ESN trajectory contains one state per non-empty timestamp bucket, not one artificial sequential state per equal-time edge. The target seed is excluded from `L_i`, so its embedding `h_i` reads the final state of the latest strictly earlier bucket only.

Example:

```text
At 10:05:  A -> B, B -> C, D -> A
At 10:06:  seed A -> E

All three 10:05 edges read C_(10:04), pool their candidates, and commit C_(10:05) once.
The 10:06 seed reads C_(10:05).
A seed at 10:05 reads C_(10:04), never another 10:05 edge.
```

### 11.3 Common tree rule

Every tree samples 64 training seeds, builds their local streams, and encodes them with the same ESN. A split family maps each seed's ESN outputs to one direct scalar difference `q_i`. At an internal node:

```text
1. create q_i using that tree family's rule and fresh random projection(s);
2. sort q_i and consider only thresholds with at least min_leaf samples
   on both sides;
4. split at a largest valid local projection gap;
5. stop when no valid split exists or the depth limit is reached.
```

Initial guard:

```text
min_leaf = 8
```

The forest never receives handcrafted graph counts or a label-derived router. The historical graph affects the model through seed-pair local link-stream sampling; every tree split consumes ESN output only.

### 11.4 Split family A: context-departure trees (100 trees)

Define the target transition from the final historical context:

\[
d_i = h_i-c_i.
\]

At a node, choose a random unit ESN direction `a` and use:

\[
q_i^{\mathrm{departure}} = a^\top d_i.
\]

This asks: **does the target transaction sharply depart from the immediately preceding local context?** It is the simple ESN baseline.

Example:

```text
the current historical state is c_i = 0.50
the target produces h_i = 2.80
the departure is h_i - c_i = +2.30

This tree regards the target as a large immediate break from the current
local context.
```

### 11.5 Split family B: transition-surprise trees (100 trees)

This family does not ask only whether the target step is large. It asks whether the target transition has appeared before in the same local ESN trajectory.

For each historical edge, form its transition:

\[
\Delta z_{i,j} = z_{i,j}-z_{i,j-1}.
\]

The target transition is `d_i = h_i-c_i`. At a node, choose a random unit direction `a` and project:

\[
s_{i,j}=a^\top\Delta z_{i,j},\qquad s_i^\ast=a^\top d_i.
\]

Let `p_i` be the smoothed empirical rank of `s_i*` among the historical values `{s_i,j}`. Use the two-sided tail surprise:

\[
q_i^{\mathrm{surprise}}
=-\log\left(2\min(p_i,1-p_i)\right).
\]

Large `q_i` means the target's **way of changing the local ESN state** lies in a tail of its own historical transition distribution.

Example:

```text
historical transitions: +0.50, -0.35, +0.50, -0.35

target transition +0.50:
  familiar local action; low transition surprise

target transition +2.30:
  unseen local action; high transition surprise
```

### 11.6 Split family C: state-recurrence trees (100 trees)

This family asks whether the state reached after the target transaction has previously occurred in the local ESN trajectory.

At a node, draw a fresh random low-dimensional projection `P`. Use:

\[
q_i^{\mathrm{recurrence}}
=
\min_{j=1}^{N_i}
\left\|P(h_i)-P(z_{i,j})\right\|_2.
\]

Small values mean that the target returns the local flow to a previously visited ESN state region. Large values mean it reaches a state region not previously visited by the local flow.

Example:

```text
historical ESN states: 0.10, 0.20, 0.75, 0.90

target state h_i = 0.15:
  close to 0.10 / 0.20; the state recurs

target state h_i = 2.30:
  far from every historical state; the state is novel
```

The projection is a computation device for high-dimensional ESN states and a source of tree diversity. This family is a testable hypothesis: ESN state drift can make recurrence unhelpful, so it must be evaluated separately rather than assumed useful.

### 11.7 300-tree forest and balanced final scoring

Construct one heterogeneous forest with a fixed, pre-declared budget:

```text
100 context-departure trees
100 transition-surprise trees
100 state-recurrence trees
```

Every tree uses the same 64-seed / local-link-stream / ESN contract. Only its split scalar differs. Seed reuse across trees is allowed.

For a transaction `e`, each 100-tree subforest produces an average path length:

```text
ell_departure(e)
ell_surprise(e)
ell_recurrence(e)
```

Do **not** average these raw path lengths: their scales can differ because the split functions differ. Instead, for each family `f`, use a label-free empirical training reference CDF `F_f` and convert shorter paths into an anomaly percentile:

\[
s_f(e)=1-F_f\left(\ell_f(e)\right).
\]

Prefer an out-of-bag training reference: when calibrating a training seed, score it only with trees that did not sample it. This avoids making in-sample paths look artificially ordinary.

The final score gives equal, pre-declared weight to all three views:

\[
S_{\mathrm{final}}(e)
=
\frac{1}{3}
\left(
s_{\mathrm{departure}}(e)
+s_{\mathrm{surprise}}(e)
+s_{\mathrm{recurrence}}(e)
\right).
\]

Equal weights are intentional: no AML label, SHAP ranking, F1, PR-AUC, Recall@K, or test result may be used to tune the 100/100/100 allocation or the aggregation weights.

At test time, process transactions chronologically. For each test seed, use only strictly earlier events to construct `L_i`, compute the same ESN trajectory and three subforest scores, then aggregate them with the formula above. Earlier unlabeled test transactions may be used as history for later test transactions; neither trees nor ESN are retrained on test labels or future transactions.

### 11.8 End-to-end worked example

```text
Build transition-surprise tree 17 of the 300-tree forest:

1. sample 64 training seeds without labels or comparison groups;
2. for each seed, sample its strictly-past 7-day k-hop L_i;
3. encode L_i into Z_i and c_i, then encode the seed into h_i;
4. choose a random ESN direction a at the root;
5. turn each seed's target transition into q_i^surprise using its own
   historical ESN transitions projected on a;
6. split the 64 q_i values at the largest valid gap;
7. at each child node, choose a fresh direction and repeat.

The same seed and ESN outputs would be handled differently by a
context-departure tree or a state-recurrence tree, but no tree receives a
handcrafted AML feature or a label.
```

## 12. Current implementation mapping

`script/dataset/linkstream/temporal_index.py::sample_temporal_subgraph` performs timestamp-strict, past-only k-hop extraction. The shared `TemporalNeighborIndex` excludes every edge at the seed timestamp, so all sampler callers share the same atomic-time semantics. `script/dataset/linkstream/collate.py::TemporalContextCollator` applies it once per seed in a batch and returns both the per-seed `contexts` list and packed `context_batch` representation.

`TemporalContextCollator` defaults to `min_context_events = 0` and exposes `sample_event_ids(...)`; it retains `N_i = 0` contexts and can extract independent seed contexts with a shared read-only thread pool. `script/model/new_model/runner.py` uses this interface directly, with a 70/30 chronological split. `LocalLinkStreamESNIF` resets a local ESN state for every context, consumes only its raw edge inputs and timestamps, and never maintains a global account state or handcrafted AML feature. Within one context, all edges sharing timestamp `t` read the same pre-bucket state `C_(t-)`; their candidates are mean-pooled and commit exactly one `C_t`. The seed reads only `C_(t-)` because same-time context edges are excluded.

The implementation uses a fixed bank of DIF-initialized Normal random ESN projections for efficient batched evaluation. The ESN recurrent matrix is initialized from the same Normal law but rescaled to `||W_res||_2 = 0.95`; with leak rate `0.20`, the sufficient contraction bound is `(1-leak) + leak * ||W_res||_2 = 0.99 < 1`. A node selects a random member of the projection bank rather than allocating a new dense projection at scoring time; this is an execution-equivalent random-representation optimization. Each subforest has a held-out train calibration set whose seed transactions were not used by any tree, so its empirical path-length reference is out-of-bag by construction.
