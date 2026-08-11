"""Prediction Market Quant Lab.

Pipeline: Research -> Backtest -> Paper Trade -> Live Trade.

Layers
------
quantlab.data        venue clients (Polymarket, Kalshi), normalization, local store
quantlab.backtest    vectorized backtest engine with realistic frictions
quantlab.strategies  signal generators that consume a normalized panel
quantlab.mispricing  fair-value estimation and the edge scanner
quantlab.cli         the QUANT LAB terminal
"""

__version__ = "0.1.0"
