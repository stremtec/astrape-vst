"""
FlowVC — 因果的ファースト音声変換 with Conditional Flow Matching.

完全因果的パイプライン:
  ソース(44.1k) → F³-Encoder(因果的, KLフリー) → z_src
    → FlowVC Converter(CFM ODE, 4-8ステップ)
    → F³-Decoder(因果的, MRFアップサンプラ) → ターゲット(44.1k)

全畳み込みが左パディングのみ（因果的）。
MioCodec非依存 — 独自エンコーダ/デコーダをスクラッチから学習。
"""

__version__ = "0.1.0"
