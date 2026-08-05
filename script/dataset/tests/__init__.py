"""Shared synthetic fixtures for the linkstream tests.

Each fixture writes tiny raw CSVs into a temp directory and runs the real
preprocessor, so the tests exercise the actual parsing + finalization path.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from linkstream import preprocessing as pp

MIDAS_PROCESSED = "midas_darpa"

# Unordered on purpose: row 0 has the latest time; rows 1 and 2 share t=10.
MIDAS_PROCESSED_CSV = """2,4,20
1,2,10
3,4,10
4,1,30
"""
MIDAS_GROUND_TRUTH_CSV = "0\n1\n0\n1\n"

PAYSIM_CSV = (
    "step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,"
    "nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud\n"
    "1,PAYMENT,100.0,C1,1000.0,900.0,M2,0.0,0.0,0,0\n"
    "1,TRANSFER,50.0,C3,50.0,0.0,C4,0.0,0.0,0,1\n"
    "2,CASH_OUT,30.0,C5,30.0,0.0,M6,100.0,0.0,1,0\n"
)

IBM_TRANS_CSV = (
    "Timestamp,From Bank,Account,To Bank,Account,Amount Received,"
    "Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering\n"
    "2022/09/01 00:20,010,8000EBD30,010,8000EBD30,3697.34,US Dollar,"
    "3697.34,US Dollar,Reinvestment,0\n"
    "2022/09/01 00:00,03208,8000F4580,001,8000F5340,0.01,US Dollar,"
    "0.01,US Dollar,Cheque,1\n"
    "2022/09/01 00:10,011,8000ECA90,011,8000ECA90,1858.96,US Dollar,"
    "1858.96,Euro,Reinvestment,0\n"
)

ELLIPTIC_CLASSES_CSV = "txId,class\na,1\nb,2\nc,3\nd,1\n"
ELLIPTIC_EDGELIST_CSV = "txId1,txId2\nx,a\na,b\nb,c\ny,d\n"
# The real features table has NO header (txId, time_step, then features).
ELLIPTIC_FEATURES_CSV = (
    "x,1,10.0\na,2,20.0\nb,3,30.0\nc,4,40.0\ny,1,50.0\nd,5,60.0\n"
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_midas_raw(root: Path) -> Path:
    _write(root / "midas_darpa" / "darpa_processed.csv", MIDAS_PROCESSED_CSV)
    _write(root / "midas_darpa" / "darpa_ground_truth.csv", MIDAS_GROUND_TRUTH_CSV)
    return root / "midas_darpa"


def write_paysim_raw(root: Path) -> Path:
    _write(root / "paysim" / "paysim.csv", PAYSIM_CSV)
    return root / "paysim"


def write_ibm_raw(root: Path, size: str = "SMALL", regime: str = "HI") -> Path:
    _write(root / "ibm_aml" / size / f"{regime}-{size.capitalize()}_Trans.csv",
           IBM_TRANS_CSV)
    return root / "ibm_aml" / size


def write_elliptic_raw(root: Path) -> Path:
    _write(root / "elliptic_pp" / "txs_classes.csv", ELLIPTIC_CLASSES_CSV)
    _write(root / "elliptic_pp" / "txs_edgelist.csv", ELLIPTIC_EDGELIST_CSV)
    _write(root / "elliptic_pp" / "txs_features.csv", ELLIPTIC_FEATURES_CSV)
    return root / "elliptic_pp"


def preprocess_midas(root: Path, out_root: Path, **kwargs) -> dict:
    write_midas_raw(root)
    return pp.preprocess_midas_darpa(
        root, out_root,
        train_ratio=kwargs.get("train_ratio", 0.6),
        val_ratio=kwargs.get("val_ratio", 0.2),
        compression=kwargs.get("compression", "zstd"),
        build_pair_index=kwargs.get("build_pair_index", False),
        overwrite=kwargs.get("overwrite", False),
    )


def preprocess_paysim(root: Path, out_root: Path, **kwargs) -> dict:
    write_paysim_raw(root)
    return pp.preprocess_paysim(
        root, out_root,
        train_ratio=kwargs.get("train_ratio", 0.6),
        val_ratio=kwargs.get("val_ratio", 0.2),
        compression=kwargs.get("compression", "zstd"),
        build_pair_index=kwargs.get("build_pair_index", False),
        overwrite=kwargs.get("overwrite", False),
    )


def preprocess_ibm(root: Path, out_root: Path, **kwargs) -> dict:
    write_ibm_raw(root, kwargs.get("size", "SMALL"), kwargs.get("regime", "HI"))
    return pp.preprocess_ibm_aml(
        root, out_root,
        size=kwargs.get("size", "SMALL"),
        regime=kwargs.get("regime", "HI"),
        train_ratio=kwargs.get("train_ratio", 0.6),
        val_ratio=kwargs.get("val_ratio", 0.2),
        compression=kwargs.get("compression", "zstd"),
        build_pair_index=kwargs.get("build_pair_index", False),
        overwrite=kwargs.get("overwrite", False),
    )


def preprocess_elliptic(root: Path, out_root: Path, **kwargs) -> dict:
    write_elliptic_raw(root)
    return pp.preprocess_elliptic_pp(
        root, out_root,
        features_mode=kwargs.get("features_mode", "none"),
        train_ratio=kwargs.get("train_ratio", 0.6),
        val_ratio=kwargs.get("val_ratio", 0.2),
        compression=kwargs.get("compression", "zstd"),
        build_pair_index=kwargs.get("build_pair_index", False),
        overwrite=kwargs.get("overwrite", False),
    )
