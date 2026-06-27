"""Backward-compat shim — Q2D2 moved to astrape/quantizer.py.

Kept so the (currently training) encoder's `from mcs_q2d2 import ...` keeps
working. New code should import from `astrape.quantizer`.
"""
from astrape.quantizer import *  # noqa: F401,F403
from astrape.quantizer import (  # explicit names used by train_mcs_q2d2.py
    Q2D2Projection, Q2D2Quantizer, compute_q2d2_perplexity,
)
