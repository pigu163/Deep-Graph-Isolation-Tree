"""Link-stream data pipeline: raw CSV -> canonical temporal link stream.

High-performance, research-oriented preprocessing and loading for temporal
graph research.  Model code is intentionally out of scope.

Package layout (under script/dataset/):

    linkstream/
        schema.py         canonical column names and file layout
        preprocessing.py  dataset-specific CSV parsers + finalization
        temporal_index.py CSR temporal neighbor index + subgraph sampling
        datasets.py       PyTorch LinkStreamDataset
        collate.py        batch dataclass, collate, loader factory
        utils.py          seeding, node mapping, splits, metadata helpers
    inspect_raw_data.py   CLI: inspect raw CSV files
    preprocess.py         CLI: convert a dataset to the canonical format
    validate_processed.py CLI: validate a processed dataset
    demo_dataloader.py    CLI: demonstrate DataLoader + throughput
    tests/                unit tests
"""

from . import (
    collate,
    datasets,
    preprocessing,
    schema,
    temporal_index,
    utils,
)

__all__ = [
    "collate",
    "datasets",
    "preprocessing",
    "schema",
    "temporal_index",
    "utils",
]
