"""
HybridVC — ハイブリッドコーデック + RAF ボコーダ音声変換パイプライン。

btrv5 コアパッケージ。以下を統合:
- MioCodec 連続潜在エンコーダ（凍結）
- CausalConvNeXt 変換器（10ブロック + クロスアテンション, ~5.4M）
- RAF BigVGAN ボコーダ（14M, 24kHz出力）
- ConvNeXt BWE（0.8M, 24k → 44.1k）
- RAF 損失（WavLM 教師 + 相対的ペアリング）

注: このパッケージは MioCodec 非因果性問題により廃止。
FlowVC パッケージ (flowvc/) が現在のメインライン。
"""

__version__ = "0.1.0"
