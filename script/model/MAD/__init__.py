"""MAD: Multi-Scale Anomaly Detection in Link Streams.

Paper-guided reproduction (WSDM '24, Bautista et al.) implemented by
porting the authors' reference code (GitLab publications1/mad, EUPL-1.2)
onto the repository's canonical link-stream pipeline.

Modules:
    probability  : slice policies + probability estimation (paper eq. 9)
    multiscale   : Haar-based multi-scale decomposition, Proposition 2,
                   standardised scoring (eq. 4-8)
    stationarity : stationarity-test window selection (paper Sec. 3.2)
    mad          : MAD object (score_edge / score_node / score_slice /
                   score_subgraph) + streaming batch scorer
    runner       : CLI producing result/MAD/... outputs
"""

from __future__ import annotations

from .mad import MAD, EdgeEventRecord, QUERY_MODES, run_edge_event_batch
from .multiscale import (
    MultiScaleResult,
    VAR_FLOOR,
    get_model_mean_var,
    haar_idwt,
    haar_wavedec,
    multiscale_approximation,
    next_power_of_two,
    order_relations,
    score_query,
)
from .probability import SlicePolicy, estimate_probabilities
from .stationarity import (
    select_window,
    stationarity_fitness,
)

__all__ = [
    "MAD",
    "EdgeEventRecord",
    "QUERY_MODES",
    "run_edge_event_batch",
    "MultiScaleResult",
    "VAR_FLOOR",
    "SlicePolicy",
    "estimate_probabilities",
    "score_query",
    "get_model_mean_var",
    "order_relations",
    "next_power_of_two",
    "haar_wavedec",
    "haar_idwt",
    "multiscale_approximation",
    "select_window",
    "stationarity_fitness",
]

__version__ = "0.1.0"
