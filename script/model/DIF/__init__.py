"""Deep Isolation Forest (DIF) — paper-guided reproduction."""

from .dif import DIF
from .random_encoder import CereEncoder, IndependentEncoder

__all__ = ["DIF", "CereEncoder", "IndependentEncoder"]
