# AGENT TASK SPECIFICATION  
## High-Performance Link-Stream Data Pipeline for Temporal Graph Research

## 0. Purpose

Implement a simple, research-oriented, high-performance data pipeline that converts the tabular datasets under this repository into a unified temporal link-stream representation.

The model itself is **out of scope for now**.

The required pipeline must support:

1. Efficiently reading large CSV files.
2. Converting each dataset into a time-ordered link stream.
3. Preserving labels and useful edge attributes.
4. Mapping raw node identifiers to compact integer IDs.
5. Saving processed data in a fast reusable format.
6. Loading link-stream events through PyTorch `Dataset` and `DataLoader`.
7. Allowing the batch size to be configured from the command line.
8. Supporting fast temporal-neighbor and temporal-subgraph queries.
9. Preventing temporal leakage.
10. Remaining simple, concise, and easy to modify for research experiments.

Do not implement the anomaly-detection model, random encoders, ESN, DIF, isolation trees, or training logic in this task.

---

# 1. Environment

## Repository root

```bash
/home/zhu/Deep Graph Isolation Tree
```

All commands should assume that the working directory is:

```bash
cd "/home/zhu/Deep Graph Isolation Tree"
```

## Python interpreter

Use this interpreter explicitly:

```bash
/home/zhu/.conda/envs/IF/bin/python
```

Example:

```bash
/home/zhu/.conda/envs/IF/bin/python scripts/preprocess.py --dataset paysim
```

Do not assume that `python` points to the correct environment.

---

# 2. Existing data layout

The current data directory is:

```text
data/
├── elliptic_pp/
│   ├── README.md
│   ├── txs_classes.csv
│   ├── txs_edgelist.csv
│   └── txs_features.csv
├── ibm_aml/
│   ├── LARGE/
│   ├── MEDIUM/
│   └── SMALL/
│       ├── HI-Small_accounts.csv
│       ├── HI-Small_Patterns.txt
│       ├── HI-Small_Trans.csv
│       ├── LI-Small_accounts.csv
│       ├── LI-Small_Patterns.txt
│       └── LI-Small_Trans.csv
├── midas_darpa/
│   ├── darpa_ground_truth.csv
│   ├── darpa_processed.csv
│   └── darpa_shape.txt
└── paysim/
    └── paysim.csv
```

For initial development, support these datasets first:

```text
midas_darpa
ibm_hi_small
ibm_li_small
paysim
elliptic_pp
```

Do not preprocess IBM Medium or Large by default.

They may be supported later through the same IBM parser by passing a size argument.

---

# 3. Scientific representation

Each temporal interaction is represented as:

\[
e_i = (t_i, u_i, v_i, x_i, y_i)
\]

where:

- `t_i`: event timestamp or discrete temporal step;
- `u_i`: source node ID;
- `v_i`: destination node ID;
- `x_i`: optional edge-feature vector;
- `y_i`: anomaly/fraud/illicit label when available.

All events must be sorted so that:

\[
t_1 \leq t_2 \leq \cdots \leq t_N
\]

The minimal link-stream tensors are:

```python
src:       torch.LongTensor       # [N]
dst:       torch.LongTensor       # [N]
ts:        torch.FloatTensor      # [N]
label:     torch.LongTensor       # [N]
edge_feat: torch.FloatTensor      # [N, d_e], possibly d_e == 0
event_id:  torch.LongTensor       # [N]
```

The original research design treats each sample as one temporal link event and expects mini-batches of ordered events:

```python
src:       LongTensor[B]
dst:       LongTensor[B]
ts:        FloatTensor[B]
edge_feat: FloatTensor[B, d_e]
label:     LongTensor[B]
event_id:  LongTensor[B]
```

The pipeline must preserve this event-level interpretation.

---

# 4. Important temporal rules

## 4.1 Never randomly shuffle the event order

The default `DataLoader` behavior must be:

```python
shuffle=False
```

Do not expose a default that silently shuffles the temporal stream.

## 4.2 No future information

Any temporal-neighbor query for event `i` at time `t_i` must only return interactions that occurred strictly before that event.

The safe event-level rule is:

```text
history indices < current event index
```

This is stronger and less ambiguous than checking only timestamps because multiple events may share the same timestamp.

## 4.3 Equal timestamps

Events with equal timestamps must preserve a deterministic stable order.

Use:

```text
primary key: timestamp
secondary key: original row index
```

Use a stable sort.

Do not reorder equal-time events nondeterministically.

## 4.4 Current-event update order

For future model integration, the intended order is:

```text
1. Query history.
2. Build the current event representation.
3. Score or predict the current event.
4. Update temporal state with the current event.
```

The data pipeline should make this order possible.

---

# 5. Technology choices

## 5.1 Preferred table-processing backend

Prefer one of the following:

1. `polars`
2. `pyarrow`
3. `pandas` only when a dataset-specific operation is simpler and memory-safe

Use `polars.scan_csv` or PyArrow streaming/batch reading for large files when practical.

Do not load a very large CSV into pandas without first considering memory usage.

Recommended strategy:

```text
CSV
→ lazy scan / chunked read
→ select only required columns
→ cast dtypes early
→ transform
→ write Parquet
```

## 5.2 Processed storage format

Use Apache Parquet as the canonical processed storage format.

Reasons:

- faster repeated loading than CSV;
- smaller disk usage;
- typed columns;
- column projection;
- compatible with Polars and PyArrow;
- easy conversion to NumPy and PyTorch.

Do not save Python pickles as the primary processed representation.

## 5.3 PyTorch loading

The runtime data layer should load processed arrays efficiently.

Acceptable approaches:

- PyArrow/Polars read followed by contiguous NumPy arrays;
- memory-mapped `.npy` arrays;
- Parquet row groups with iterable loading.

For the first implementation, prefer the simplest reliable option:

```text
Parquet preprocessing output
→ load required columns once
→ contiguous NumPy arrays
→ torch.from_numpy without unnecessary copies
```

If a dataset is too large for RAM, add an optional iterable/chunked mode without making the base implementation complicated.

---

# 6. Required project structure

Create a compact structure similar to:

```text
src/
└── linkstream/
    ├── __init__.py
    ├── schema.py
    ├── preprocessing.py
    ├── datasets.py
    ├── temporal_index.py
    ├── collate.py
    └── utils.py

scripts/
├── inspect_raw_data.py
├── preprocess.py
├── validate_processed.py
└── demo_dataloader.py

tests/
├── test_preprocessing.py
├── test_temporal_index.py
└── test_dataloader.py
```

Do not create a large framework.

Keep the number of modules small and responsibilities clear.

---

# 7. Canonical processed schema

Every processed event table must contain:

| Column | Type | Required | Meaning |
|---|---:|---:|---|
| `event_id` | `int64` | yes | Position after stable temporal sorting |
| `timestamp` | `float64` or `int64` | yes | Numeric event time |
| `src` | `int64` | yes | Compact source node ID |
| `dst` | `int64` | yes | Compact destination node ID |
| `label` | `int8` | yes | `0`, `1`, or `-1` for unavailable |
| `label_available` | `bool` | yes | Whether label is known |
| `original_row_id` | `int64` | yes | Row ID before sorting |
| `edge_type` | categorical/string | optional | Transaction or interaction type |
| `amount` | `float32` | optional | Transaction amount |
| feature columns | `float32` | optional | Dataset-specific numerical features |

Use `label = -1` for unknown or unavailable labels.

Do not convert unknown labels to normal.

---

# 8. Output layout

Each processed dataset should be written to:

```text
data/processed/<dataset_name>/
├── events.parquet
├── nodes.parquet
├── metadata.json
├── splits.json
└── temporal_index/
    ├── src_offsets.npy
    ├── src_event_ids.npy
    ├── dst_offsets.npy
    ├── dst_event_ids.npy
    ├── node_offsets.npy
    ├── node_event_ids.npy
    └── timestamps.npy
```

Optional pair index:

```text
temporal_index/
├── pair_keys.npy
├── pair_offsets.npy
└── pair_event_ids.npy
```

Only build the pair index when requested:

```bash
--build-pair-index
```

Pair indexing may consume substantial memory on large datasets.

---

# 9. Node mapping

Convert arbitrary raw node identifiers to contiguous integers:

```text
0, 1, 2, ..., num_nodes - 1
```

Save:

```text
nodes.parquet
```

with:

| Column | Meaning |
|---|---|
| `node_id` | compact integer ID |
| `raw_id` | original identifier |
| `node_type` | optional entity type |
| `first_event_id` | first observed event |
| `first_timestamp` | first observed time |

Mapping must be deterministic for a fixed input.

Recommended deterministic rule:

```text
Assign IDs in order of first appearance in the stable time-sorted stream.
```

Do not use Python's unordered set iteration to assign IDs.

---

# 10. Dataset-specific conversion

## 10.1 MIDAS-DARPA

Raw files:

```text
data/midas_darpa/darpa_processed.csv
data/midas_darpa/darpa_ground_truth.csv
data/midas_darpa/darpa_shape.txt
```

Expected logical mapping:

```text
src, dst, timestamp
```

Ground-truth labels are aligned by row.

Tasks:

1. Inspect the exact delimiter and header state.
2. Read events and labels.
3. Assert that the number of labels equals the number of events.
4. Add `original_row_id`.
5. Stable-sort by timestamp and original row ID.
6. Map source and destination identifiers to compact node IDs.
7. Save the unified event table.

Canonical mapping:

```text
timestamp = raw timestamp
src       = raw source
dst       = raw destination
label     = aligned ground-truth label
```

No rich edge features are required initially.

---

## 10.2 IBM AML

Initial files:

```text
data/ibm_aml/SMALL/HI-Small_Trans.csv
data/ibm_aml/SMALL/LI-Small_Trans.csv
```

Also retain but do not require for the first event pipeline:

```text
*_accounts.csv
*_Patterns.txt
```

The exact headers must be inspected before implementation.

Expected transaction concepts include:

```text
Timestamp
From Bank
source account
To Bank
destination account
Amount Received
Receiving Currency
Amount Paid
Payment Currency
Payment Format
Is Laundering
```

Node IDs must include both bank and account:

```python
src_raw = from_bank + ":" + source_account
dst_raw = to_bank + ":" + destination_account
```

Do not identify nodes using the account field alone.

Canonical mapping:

```text
timestamp = parsed Timestamp
src       = (From Bank, source account)
dst       = (To Bank, destination account)
label     = Is Laundering
amount    = Amount Paid
edge_type = Payment Format
```

Optional features:

```text
Amount Paid
Amount Received
same_bank
same_currency
payment currency
receiving currency
```

For the first implementation, keep preprocessing conservative:

- preserve raw categorical columns in Parquet;
- create only obvious numerical features;
- do not design a model-specific feature engineering pipeline.

The preprocessing CLI must support:

```bash
--dataset ibm_aml
--size SMALL
--regime HI
```

and:

```bash
--dataset ibm_aml
--size SMALL
--regime LI
```

Do not process `MEDIUM` or `LARGE` unless explicitly requested.

---

## 10.3 PaySim

Raw file:

```text
data/paysim/paysim.csv
```

Expected fields:

```text
step
type
amount
nameOrig
oldbalanceOrg
newbalanceOrig
nameDest
oldbalanceDest
newbalanceDest
isFraud
isFlaggedFraud
```

Canonical mapping:

```text
timestamp = step
src       = nameOrig
dst       = nameDest
label     = isFraud
amount    = amount
edge_type = type
```

Preserve node prefixes such as:

```text
C...
M...
```

Do not strip them.

Recommended optional numerical features:

```text
amount
oldbalanceOrg
newbalanceOrig
oldbalanceDest
newbalanceDest
```

Do not use `isFlaggedFraud` as the ground-truth label.

Keep it only as an optional auxiliary column.

---

## 10.4 Elliptic++

Raw files:

```text
data/elliptic_pp/txs_classes.csv
data/elliptic_pp/txs_edgelist.csv
data/elliptic_pp/txs_features.csv
```

This dataset differs conceptually from the account transaction datasets.

Here:

```text
node = Bitcoin transaction
edge = money flow from an upstream transaction to a downstream transaction
```

The preprocessing script must inspect the actual headers and README.

Expected workflow:

1. Read transaction features.
2. identify the transaction ID column;
3. identify the transaction time-step column;
4. read transaction classes;
5. read directed transaction edges;
6. join the destination transaction metadata onto each edge;
7. define event timestamp as the destination transaction's time step;
8. define event label from the destination transaction's node label.

Canonical event mapping:

```text
src       = upstream transaction ID
dst       = downstream transaction ID
timestamp = time step of dst
label     = class of dst
```

Label conversion:

```text
illicit -> 1
licit   -> 0
unknown -> -1
```

and:

```text
label_available = label != -1
```

The exact raw symbols may be numeric or strings. Inspect and document them.

Important:

```text
The Elliptic++ label is originally a node label.
The event-level label is derived from the destination node.
```

Record this derivation in `metadata.json`.

Do not use future node features or graph statistics that depend on edges occurring after the current event.

For the first implementation, include only transaction features that are already supplied per node and can be joined safely. Clearly distinguish:

```text
raw local features
aggregated features
```

If their temporal validity is uncertain, make feature inclusion configurable and default to a minimal structure-only event stream.

Recommended flag:

```bash
--elliptic-features none
```

Possible later options:

```text
none
local
all
```

Default:

```text
none
```

---

# 11. Temporal neighbor index

## 11.1 Objective

While iterating through the link stream, downstream code must be able to efficiently query:

- previous outgoing events of a source node;
- previous incoming events of a destination node;
- previous incident events of either endpoint;
- previous interactions of a directed pair;
- recent temporal neighbors before a given event;
- a local temporal subgraph before a given event.

Do not reconstruct these relations by filtering the full event table for every sample.

## 11.2 Base index representation

Use compressed flat arrays instead of Python lists of lists for the saved index.

Recommended node-event CSR-style representation:

```text
node_offsets:   int64[num_nodes + 1]
node_event_ids: int64[2 * num_events]
```

Each event appears once for its source and once for its destination.

For directed queries, also store:

```text
src_offsets:    int64[num_nodes + 1]
src_event_ids:  int64[num_events]

dst_offsets:    int64[num_nodes + 1]
dst_event_ids:  int64[num_events]
```

The event IDs inside each node segment must be sorted in temporal/event order.

Save a global timestamp array:

```text
timestamps: float64[num_events]
```

## 11.3 Query semantics

For current event ID `i`, a temporal-history query must only return:

```text
event_id < i
```

Even when earlier and current events share the same timestamp.

Support:

```python
get_past_out_events(node_id, event_id, max_events=None, time_window=None)
get_past_in_events(node_id, event_id, max_events=None, time_window=None)
get_past_incident_events(node_id, event_id, max_events=None, time_window=None)
get_past_pair_events(src, dst, event_id, max_events=None, time_window=None)
```

Return event IDs, not copied event records.

The caller can gather source, destination, timestamps, labels, and features using those IDs.

## 11.4 Fast boundary search

Use binary search over sorted event IDs or timestamps.

Suitable tools:

```python
numpy.searchsorted
```

Do not linearly scan an entire node history for every query.

## 11.5 Pair index

A directed pair `(src, dst)` may be encoded as:

```python
pair_key = src * num_nodes + dst
```

Use `int64` and check for overflow, although ordinary dataset sizes should be safe.

Build a compact sorted pair index only when requested.

Avoid a huge Python dictionary containing Python lists for all pairs in the persistent representation.

A temporary dictionary during preprocessing is acceptable only if memory usage is reasonable.

A scalable alternative is:

1. create arrays of `pair_key` and `event_id`;
2. stable-sort by `(pair_key, event_id)`;
3. compute unique pair keys and offsets;
4. save compact arrays.

---

# 12. Temporal subgraph sampling

Implement a simple temporal subgraph sampler around the current event `(u, v, t)`.

Required initial behavior:

```python
sample_temporal_subgraph(
    event_id: int,
    num_hops: int = 1,
    max_neighbors: int = 20,
    time_window: float | None = None,
    direction: str = "both",
) -> TemporalSubgraph
```

For one hop:

1. Start from `src[event_id]` and `dst[event_id]`.
2. Query only past incident events.
3. Keep at most `max_neighbors` most recent events per seed node.
4. Return the unique involved nodes and selected historical event IDs.
5. Exclude the current event.

For multiple hops:

1. Expand from nodes reached in the previous hop.
2. Never include an event with `event_id >= current_event_id`.
3. Deduplicate event IDs and node IDs.
4. Apply explicit caps to prevent uncontrolled expansion.

Return a simple dataclass:

```python
@dataclass
class TemporalSubgraph:
    node_ids: torch.LongTensor
    event_ids: torch.LongTensor
    src: torch.LongTensor
    dst: torch.LongTensor
    ts: torch.Tensor
    edge_feat: torch.Tensor | None
```

Do not initially create PyTorch Geometric `Data` objects unless the repository already depends on PyG.

Keep the core pipeline framework-neutral.

---

# 13. Dataset and DataLoader API

## 13.1 Event dataset

Implement a map-style dataset:

```python
class LinkStreamDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        processed_dir: str | Path,
        split: str = "all",
        include_edge_features: bool = True,
    ):
        ...
```

Required:

```python
len(dataset) == number of selected events
```

Each item should be lightweight:

```python
{
    "event_id": int,
    "src": int,
    "dst": int,
    "ts": float,
    "label": int,
    "label_available": bool,
    "edge_feat": Tensor[d_e],
}
```

Avoid creating a new large Python object graph during initialization.

## 13.2 Batch object

Use a dataclass or `NamedTuple`:

```python
@dataclass
class LinkStreamBatch:
    event_id: torch.LongTensor
    src: torch.LongTensor
    dst: torch.LongTensor
    ts: torch.Tensor
    label: torch.LongTensor
    label_available: torch.BoolTensor
    edge_feat: torch.Tensor | None
```

## 13.3 Collate function

Implement a clear collate function:

```python
def collate_link_stream(samples: list[dict]) -> LinkStreamBatch:
    ...
```

Keep events in the exact incoming order.

Do not sort inside the collate function unless explicitly documented and tested.

## 13.4 DataLoader factory

Implement:

```python
def make_link_stream_loader(
    dataset: LinkStreamDataset,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    ...
```

Defaults for temporal research:

```python
shuffle=False
drop_last=False
num_workers=0
```

`num_workers=0` should be the safe default because:

- order is easier to reason about;
- arrays are already loaded efficiently;
- future stateful temporal models may not be compatible with arbitrary worker behavior.

Allow override from CLI.

---

# 14. Optional batch-level history output

Do not make temporal subgraph extraction mandatory for every batch in the base loader.

It may be expensive and model-specific.

Provide one optional wrapper:

```python
class TemporalContextCollator:
    def __init__(
        self,
        temporal_index: TemporalNeighborIndex,
        num_hops: int,
        max_neighbors: int,
        time_window: float | None,
    ):
        ...
```

It may return:

```python
{
    "events": LinkStreamBatch,
    "contexts": list[TemporalSubgraph],
}
```

Keep the default loader event-only.

This separation allows:

- fast sequential event loading;
- configurable context extraction;
- fair timing experiments;
- model-independent preprocessing.

---

# 15. Train/validation/test splits

Use chronological splits.

Default:

```text
train: first 60%
validation: next 20%
test: final 20%
```

Expose:

```bash
--train-ratio 0.6
--val-ratio 0.2
```

The remainder is test.

Save split boundaries in:

```text
splits.json
```

Example:

```json
{
  "train": [0, 3600000],
  "val": [3600000, 4800000],
  "test": [4800000, 6000000]
}
```

Use contiguous event-ID ranges.

Do not randomly stratify temporal events.

---

# 16. Random seeds

Every executable script must accept:

```bash
--seed 42
```

Create one utility:

```python
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

For deterministic node mapping and preprocessing, randomness should normally not be required.

Seeds matter for:

- optional neighbor subsampling;
- future training;
- randomized validation tests;
- benchmark repetitions.

When sampling neighbors, accept a seeded `numpy.random.Generator` or `torch.Generator`.

Do not rely on implicit global randomness inside core index methods.

---

# 17. Command-line interfaces

Use `argparse`.

## 17.1 Inspect raw files

```bash
/home/zhu/.conda/envs/IF/bin/python scripts/inspect_raw_data.py \
  --dataset paysim \
  --num-rows 5
```

The script should print:

- detected path;
- file size;
- columns;
- inferred dtypes;
- first rows;
- null counts for sampled/full columns;
- estimated row count when cheap;
- label values and counts when available.

## 17.2 Preprocess

Examples:

```bash
/home/zhu/.conda/envs/IF/bin/python scripts/preprocess.py \
  --dataset midas_darpa \
  --seed 42
```

```bash
/home/zhu/.conda/envs/IF/bin/python scripts/preprocess.py \
  --dataset ibm_aml \
  --size SMALL \
  --regime HI \
  --seed 42
```

```bash
/home/zhu/.conda/envs/IF/bin/python scripts/preprocess.py \
  --dataset paysim \
  --seed 42
```

```bash
/home/zhu/.conda/envs/IF/bin/python scripts/preprocess.py \
  --dataset elliptic_pp \
  --elliptic-features none \
  --seed 42
```

Required arguments:

```text
--dataset
--data-root
--output-root
--seed
```

Useful optional arguments:

```text
--overwrite
--build-pair-index
--train-ratio
--val-ratio
--chunk-size
--compression
--size
--regime
--elliptic-features
```

Defaults:

```text
--data-root "/home/zhu/Deep Graph Isolation Tree/data"
--output-root "/home/zhu/Deep Graph Isolation Tree/data/processed"
--seed 42
--compression zstd
```

## 17.3 Validate

```bash
/home/zhu/.conda/envs/IF/bin/python scripts/validate_processed.py \
  --dataset ibm_hi_small
```

Validation must check:

- nondecreasing timestamps;
- contiguous event IDs;
- source and destination IDs in range;
- expected label values;
- split boundaries;
- temporal-index offsets;
- sorted event IDs within every index segment;
- no temporal query returns the current or future event.

## 17.4 DataLoader demo

```bash
/home/zhu/.conda/envs/IF/bin/python scripts/demo_dataloader.py \
  --dataset paysim \
  --batch-size 1024 \
  --num-workers 0 \
  --seed 42
```

The script should print:

```text
dataset name
number of events
number of nodes
edge feature dimension
batch tensor shapes
first timestamp
last timestamp
label distribution
loading throughput
```

Optional temporal-context demo:

```bash
--with-context
--num-hops 1
--max-neighbors 20
```

---

# 18. Metadata

Write `metadata.json` for every dataset.

Required fields:

```json
{
  "dataset_name": "paysim",
  "num_events": 0,
  "num_nodes": 0,
  "num_labeled_events": 0,
  "num_positive_labels": 0,
  "timestamp_min": 0,
  "timestamp_max": 0,
  "edge_feature_names": [],
  "raw_files": [],
  "node_mapping_rule": "first appearance after stable temporal sort",
  "event_label_definition": "",
  "notes": []
}
```

For Elliptic++, explicitly note:

```text
Event labels are derived from destination transaction node labels.
```

For IBM AML, explicitly note:

```text
Node IDs combine bank ID and account ID.
```

---

# 19. Performance requirements

The implementation should be efficient without becoming complicated.

## 19.1 Avoid

- row-wise pandas `apply`;
- Python loops over millions of rows when vectorized operations are available;
- repeatedly filtering the full event table for neighbor lookup;
- storing every event as a Python dictionary in memory;
- NetworkX for the main temporal index;
- materializing all temporal subgraphs in advance;
- duplicate copies of large feature matrices;
- hidden sorting inside every DataLoader epoch.

## 19.2 Prefer

- Polars expressions;
- PyArrow arrays;
- NumPy contiguous arrays;
- stable vectorized sorting;
- compact integer dtypes where safe;
- `numpy.searchsorted`;
- CSR-style flat temporal indexes;
- lazy or chunked CSV reading;
- Parquet with Zstandard compression;
- `torch.from_numpy`.

## 19.3 Benchmark

The demo script should report approximate event throughput:

```text
events/second
```

Measure separately:

1. base event loading;
2. event loading with temporal context extraction.

Do not claim GPU acceleration for CSV parsing or neighbor indexing unless it is actually implemented and measured.

---

# 20. Simplicity requirements

This is research code.

The code must be:

- concise;
- readable;
- modular enough for experiments;
- easy to debug;
- easy to alter;
- documented with clear English comments;
- typed where useful;
- free of unnecessary abstraction.

Do not add:

- configuration frameworks such as Hydra;
- dependency injection frameworks;
- experiment tracking systems;
- databases;
- web APIs;
- distributed training;
- plugin architectures;
- deeply nested class hierarchies;
- automatic downloading;
- model implementations.

Use:

- standard Python;
- `argparse`;
- `dataclasses`;
- `pathlib`;
- NumPy;
- PyTorch;
- Polars or PyArrow;
- small JSON metadata files.

---

# 21. Dependency handling

Before coding, inspect the environment:

```bash
/home/zhu/.conda/envs/IF/bin/python -c \
"import torch, numpy; print(torch.__version__, numpy.__version__)"
```

Check optional libraries:

```bash
/home/zhu/.conda/envs/IF/bin/python -c \
"import polars, pyarrow; print(polars.__version__, pyarrow.__version__)"
```

If `polars` or `pyarrow` is missing, do not silently install packages.

Report the missing dependency and provide the exact user-space installation command, for example:

```bash
/home/zhu/.conda/envs/IF/bin/python -m pip install polars pyarrow
```

Do not use `sudo`.

Create or update a minimal dependency file only if appropriate:

```text
requirements-data.txt
```

Suggested contents:

```text
numpy
torch
polars
pyarrow
```

Do not pin versions unless compatibility requires it.

---

# 22. Tests

## 22.1 Preprocessing tests

Use small synthetic CSV fixtures.

Test:

- timestamp parsing;
- stable sorting;
- deterministic node mapping;
- label alignment;
- unknown-label handling;
- IBM bank-account concatenation;
- Elliptic destination-label join.

## 22.2 Temporal-index tests

Construct a tiny stream:

```text
event 0: t=1, 0 -> 1
event 1: t=2, 1 -> 2
event 2: t=2, 0 -> 2
event 3: t=3, 2 -> 0
```

Verify:

```text
history(event 0) is empty
history of node 0 before event 2 contains event 0 only
history before event 3 never contains event 3
equal-timestamp ordering follows event ID
```

Test:

- incoming history;
- outgoing history;
- incident history;
- `max_events`;
- `time_window`;
- pair history;
- one-hop temporal subgraph;
- no future leakage.

## 22.3 DataLoader tests

Verify:

- configurable batch size;
- no shuffling;
- final incomplete batch is retained by default;
- event IDs remain increasing across batches;
- tensor dtypes;
- tensor shapes;
- labels remain aligned.

---

# 23. Acceptance criteria

The task is complete when all of the following work.

## 23.1 Preprocessing

At least these commands succeed:

```bash
/home/zhu/.conda/envs/IF/bin/python scripts/preprocess.py \
  --dataset midas_darpa
```

```bash
/home/zhu/.conda/envs/IF/bin/python scripts/preprocess.py \
  --dataset ibm_aml --size SMALL --regime HI
```

```bash
/home/zhu/.conda/envs/IF/bin/python scripts/preprocess.py \
  --dataset ibm_aml --size SMALL --regime LI
```

```bash
/home/zhu/.conda/envs/IF/bin/python scripts/preprocess.py \
  --dataset paysim
```

```bash
/home/zhu/.conda/envs/IF/bin/python scripts/preprocess.py \
  --dataset elliptic_pp --elliptic-features none
```

## 23.2 DataLoader

This must work:

```python
dataset = LinkStreamDataset(
    processed_dir="data/processed/paysim",
    split="train",
)

loader = make_link_stream_loader(
    dataset,
    batch_size=1024,
    num_workers=0,
)

batch = next(iter(loader))

assert batch.src.shape == (1024,)
assert batch.dst.shape == (1024,)
assert batch.ts.shape == (1024,)
assert batch.label.shape == (1024,)
```

## 23.3 Temporal index

This must work:

```python
index = TemporalNeighborIndex.load(
    "data/processed/paysim/temporal_index"
)

history = index.get_past_incident_events(
    node_id=int(batch.src[0]),
    event_id=int(batch.event_id[0]),
    max_events=20,
)

assert (history < batch.event_id[0].item()).all()
```

## 23.4 Reproducibility

Running preprocessing twice with identical inputs and seed must produce:

- the same event ordering;
- the same node IDs;
- the same split boundaries;
- identical core arrays.

## 23.5 Documentation

Add a concise repository document:

```text
DATA_PIPELINE.md
```

It should explain:

- processed format;
- dataset-specific mappings;
- preprocessing commands;
- DataLoader usage;
- temporal-neighbor queries;
- temporal leakage rules.

---

# 24. Recommended implementation order

Follow this order.

## Phase 1: Inspect

1. Inspect installed packages.
2. Inspect actual CSV headers and dtypes.
3. Record any deviations from expected schemas.
4. Do not guess silently.

## Phase 2: Core schema

1. Define canonical event schema.
2. Define metadata schema.
3. Define deterministic node mapping.
4. Define chronological splits.

## Phase 3: Easy datasets

1. Implement MIDAS-DARPA.
2. Implement PaySim.
3. Validate event and label alignment.

## Phase 4: IBM AML

1. Implement HI-Small.
2. Reuse the same parser for LI-Small.
3. Keep Medium/Large parameterized but do not run by default.

## Phase 5: Elliptic++

1. Inspect transaction ID, class, time-step, and edge columns.
2. Implement joins.
3. Default to structure-only.
4. Preserve unknown labels.

## Phase 6: Temporal index

1. Build source CSR.
2. Build destination CSR.
3. Build incident-node CSR.
4. Add binary-search queries.
5. Add optional pair CSR.
6. Add temporal-subgraph sampling.

## Phase 7: PyTorch API

1. Implement `LinkStreamDataset`.
2. Implement batch dataclass.
3. Implement collate function.
4. Implement loader factory.
5. Add optional context collator.

## Phase 8: Validation and tests

1. Run unit tests.
2. Validate all processed datasets.
3. Benchmark event loading.
4. Document commands and observed dataset statistics.

---

# 25. Agent behavior

Before changing code:

1. Inspect the repository.
2. Inspect all relevant raw file headers.
3. Check existing coding conventions.
4. Check installed dependencies.
5. State a short implementation plan.

During implementation:

1. Make small, logically grouped changes.
2. Run tests after each major component.
3. Do not delete or modify raw data.
4. Do not process Medium/Large unless explicitly requested.
5. Do not add model code.
6. Do not introduce temporal leakage.
7. Keep comments in clear English.

After implementation, report:

1. Files created or modified.
2. Exact commands run.
3. Test results.
4. Processed dataset statistics.
5. Approximate preprocessing and loading throughput.
6. Any unresolved schema ambiguity.
7. Any optional improvement deliberately deferred.

---

# 26. Design rationale

A link stream is naturally represented as time-stamped directed interactions `(t, u, v)`. The MAD formulation also treats link streams as directed temporal relations and permits time-restricted subgraph queries. The TGN formulation treats continuous-time dynamic graphs as ordered interaction events and defines temporal neighborhoods over time intervals.

This project therefore needs two distinct but connected representations:

```text
Sequential event representation
    for efficient ordered mini-batch processing

Temporal neighbor index
    for efficient historical neighborhood and subgraph retrieval
```

Do not replace the event stream with precomputed static snapshots.

Snapshots may be generated later as a view over the event stream, but they must not be the canonical stored representation.

The canonical data object remains an ordered sequence of temporal link events.
