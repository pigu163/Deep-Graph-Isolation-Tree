# Link-Stream Data Pipeline

Research-oriented pipeline that converts the tabular datasets in `data/`
into a unified, time-ordered **link-stream** representation for temporal
graph research (anomaly detection in link streams). Model code is out of
scope here.

All pipeline code lives in `script/dataset/`:

```text
script/dataset/
├── linkstream/
│   ├── schema.py          canonical column names + file layout
│   ├── preprocessing.py   dataset parsers (DARPA / PaySim / IBM / Elliptic++)
│   ├── temporal_index.py  CSR temporal neighbor index + subgraph sampling
│   ├── datasets.py        torch.utils.data.Dataset
│   ├── collate.py         batch dataclass, collate, loader factory
│   └── utils.py           seeds, node mapping, splits, metadata
├── inspect_raw_data.py
├── preprocess.py
├── validate_processed.py
├── demo_dataloader.py
└── tests/                 unittest suites
```

Use the project interpreter explicitly:

```bash
/home/zhu/.conda/envs/GRAG/bin/python ...
```

## 1. Processed format

Each event is a temporal interaction `e_i = (t_i, u_i, v_i, x_i, y_i)`.
Output per dataset under `data/processed/<dataset>/`:

```text
events.parquet        ordered event table (canonical schema below)
nodes.parquet         raw_id -> compact node_id mapping
metadata.json         dataset statistics and provenance
splits.json           chronological train/val/test event-id ranges
temporal_index/
    src_offsets.npy  src_event_ids.npy
    dst_offsets.npy  dst_event_ids.npy
    node_offsets.npy node_event_ids.npy
    timestamps.npy
    pair_keys.npy pair_offsets.npy pair_event_ids.npy   # only with --build-pair-index
```

Canonical `events.parquet` columns:

| Column | Type | Meaning |
|---|---:|---|
| `event_id` | int64 | position after stable temporal sort |
| `timestamp` | float64 | numeric event time |
| `src` / `dst` | int64 | compact node ids |
| `label` | int8 | 0 / 1 / -1 (unknown) |
| `label_available` | bool | whether the label is known |
| `original_row_id` | int64 | row id in the raw file before sorting |
| `edge_type` | string | optional transaction/interaction type |
| `amount` | float32 | optional amount |
| `f_*` | float32 | optional numeric edge features |
| dataset extras | - | e.g. `is_flagged_fraud` (PaySim), currencies (IBM) |

Node ids are compact integers `0..num_nodes-1`, assigned in **order of
first appearance in the stable time-sorted stream** (deterministic for a
fixed input).

Events are sorted by `(timestamp, original_row_id)` — a stable sort whose
secondary key makes equal-time ordering fully deterministic.

## 2. Dataset-specific mappings

| Dataset | timestamp | src | dst | label | amount | edge_type |
|---|---|---|---|---|---|---|
| `midas_darpa` | raw time | raw source | raw dest | ground truth (by row) | - | - |
| `paysim` | `step` | `nameOrig` | `nameDest` | `isFraud` | `amount` | `type` |
| `ibm_aml` | parsed `Timestamp` (epoch s) | `From Bank:From Account` | `To Bank:To Account` | `Is Laundering` | `Amount Paid` | `Payment Format` |
| `elliptic_pp` | dst tx `time_step` | upstream tx id | downstream tx id | **dst** tx class | - | - |

Notes:

- **IBM**: node ids combine bank id and account id (`bank:account`); the raw
  CSV has two columns both named `Account`, renamed to `From Account` /
  `To Account` on read. Timestamps are parsed as `%Y/%m/%d %H:%M` and
  converted to unix-epoch seconds.
- **Elliptic++**: node = Bitcoin transaction, edge = directed money flow.
  Event timestamp and label are those of the **destination** transaction
  (`class`: 1→1 illicit, 2→0 licit, else → -1 unknown). The huge
  `txs_features.csv` is lazy-scanned; default `--elliptic-features none`
  keeps a structure-only stream, `all` additionally joins raw per-node
  features (whose temporal validity is not guaranteed).
- **PaySim**: `isFlaggedFraud` is kept only as an auxiliary column, never
  used as the label. Node prefixes (`C...`/`M...`) are preserved.

## 3. Preprocessing commands

```bash
cd "/home/zhu/Deep Graph Isolation Tree"

/home/zhu/.conda/envs/GRAG/bin/python script/dataset/preprocess.py --dataset midas_darpa
/home/zhu/.conda/envs/GRAG/bin/python script/dataset/preprocess.py --dataset ibm_aml --size SMALL --regime HI
/home/zhu/.conda/envs/GRAG/bin/python script/dataset/preprocess.py --dataset ibm_aml --size SMALL --regime LI
/home/zhu/.conda/envs/GRAG/bin/python script/dataset/preprocess.py --dataset paysim
/home/zhu/.conda/envs/GRAG/bin/python script/dataset/preprocess.py --dataset elliptic_pp --elliptic-features none
```

Useful flags: `--overwrite`, `--build-pair-index`, `--train-ratio`,
`--val-ratio`, `--compression` (default `zstd`), `--seed` (default 42).

Validate a processed dataset:

```bash
/home/zhu/.conda/envs/GRAG/bin/python script/dataset/validate_processed.py --dataset paysim
```

Inspect raw files before preprocessing:

```bash
/home/zhu/.conda/envs/GRAG/bin/python script/dataset/inspect_raw_data.py --dataset paysim --num-rows 5
```

## 4. DataLoader usage

```python
from linkstream.datasets import LinkStreamDataset
from linkstream.collate import make_link_stream_loader

dataset = LinkStreamDataset("data/processed/paysim", split="train")  # or "all"
loader = make_link_stream_loader(dataset, batch_size=1024, num_workers=0)

batch = next(iter(loader))
# batch.event_id/src/dst/ts/label/label_available, batch.edge_feat (or None)
```

Defaults are temporal-research-safe: `shuffle=False`, `drop_last=False`,
`num_workers=0`. The final incomplete batch is kept.

Demo with throughput report:

```bash
/home/zhu/.conda/envs/GRAG/bin/python script/dataset/demo_dataloader.py \
  --dataset paysim --batch-size 1024 --with-context --num-hops 1 --max-neighbors 20
```

## 5. Temporal-neighbor queries

```python
from linkstream.temporal_index import TemporalNeighborIndex

index = TemporalNeighborIndex.load("data/processed/paysim/temporal_index")

out      = index.get_past_out_events(node_id, event_id, max_events=20)
inn      = index.get_past_in_events(node_id, event_id, max_events=20)
incident = index.get_past_incident_events(node_id, event_id, max_events=20)
pair     = index.get_past_pair_events(src, dst, event_id)  # needs --build-pair-index
```

All queries return **event ids** (never copied records) and only ever return
events with `event_id < current_event_id`. The index is CSR-style flat
arrays; lookups are `numpy.searchsorted` binary searches over sorted
segments, never full-table scans.

One-hop / multi-hop temporal subgraphs:

```python
from linkstream.temporal_index import sample_temporal_subgraph

sg = sample_temporal_subgraph(
    index, event_id, src_arr=src, dst_arr=dst, ts_arr=ts, edge_feat=edge_feat,
    num_hops=1, max_neighbors=20, direction="both")
# sg: node_ids, event_ids, src, dst, ts, edge_feat (torch tensors)
```

or use `TemporalContextCollator` to attach a subgraph to every event in a
batch (`{"events": batch, "contexts": [...]}`).

## 6. Temporal leakage rules

1. **Never shuffle** the event stream (`shuffle=False`).
2. **Strictly past**: a history query for event `i` returns only `event_id < i`,
   which is stronger than timestamp comparison when events share a timestamp.
3. **Stable equal-time order**: sort by `(timestamp, original_row_id)`.
4. **Query-before-update**: models must query history, build the current
   representation, score, and only then update temporal state — the pipeline
   supports this order (see `contextual_temporal_dif_design.md`).
5. Splits are chronological and contiguous; labels never leak across splits.

## 7. Environment notes

- The GRAG conda environment has `torch 2.5.1`, `numpy 1.26.4`,
  `polars 1.39.3`; **`pyarrow` is not installed**. The pipeline uses Polars
  only (Polars writes/reads Parquet natively), so pyarrow is optional.
- The `elliptic_pp/*.csv` files in this checkout are **Git LFS pointers**;
  materialize them (`git lfs pull`) before preprocessing that dataset. The
  preprocessor raises a clear error otherwise.
- Run the tests with:

```bash
cd "/home/zhu/Deep Graph Isolation Tree"
/home/zhu/.conda/envs/GRAG/bin/python -m unittest discover -s script/dataset/tests -v
```
