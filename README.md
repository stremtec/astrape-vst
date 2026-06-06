# Astrape VC ⚡

> **Αστραπή** (アストラペー) — ギリシャ語で「稲妻」

Conditional Flow Matching によるリアルタイムニューラル音声変換。  
完全因果的パイプライン。44.1kHz。MioCodec非依存。

```
ソース(44.1k) → F³-Encoder(因果的, KLフリー) → z_src
  → FlowVC Converter(CFM ODE, 4-8ステップ)
  → F³-Decoder(因果的, MRFアップサンプラ) → ターゲット(44.1k)
```

## アーキテクチャ

| コンポーネント | パラメータ | 説明 |
|-----------|:------:|-------------|
| F³-Encoder | 26.5M | 6段因果的ConvNeXt v2, KLフリー, ノイズ正則化 |
| VectorFieldNet | 37.2M | 12ブロックCFM変換器, AdaLN-Zero, クロスアテンション |
| F³-Decoder | 27.9M | 6段MRFアップサンプラ, FiLM条件付け |
| **合計** | **91.6M** | 完全因果的, ストリーミング対応 |

## 主要特徴

- **因果的ファースト**: 全畳み込みが左パディングのみ — 未来の情報漏洩なし
- **Flow Matching**: OTパスによる条件付きフローマッチング, 4ステップEulerソルバ
- **KLフリー**: VQなし, コードブック崩壊なし, 連続潜在空間
- **ConvNeXt v2**: GRN, LayerScale, 逆ボトルネックを全体に採用
- **44.1kHzネイティブ**: 帯域拡張不要
- **Zero-init全般**: 初期化時に恒等写像, 安定した学習

## クイックスタート

```bash
# 形状テスト
python3 -c "
from flowvc.encoder import make_encoder
from flowvc.converter import make_vector_field_net, solve_cfm_euler
from flowvc.decoder import F3Decoder
from flowvc.config import DecoderConfig
import torch

encoder = make_encoder()
vfn = make_vector_field_net()
decoder = F3Decoder(DecoderConfig())

B, T_lat = 2, 50
wav = torch.randn(B, 1, T_lat * 1764)  # 2秒 @ 44.1kHz
spk = torch.randn(B, 192)
prompt = torch.randn(B, 4, 192)
prosody = torch.randn(B, T_lat, 3)

z = encoder.encode(wav)
z_tgt = solve_cfm_euler(vfn, z, spk, prompt, prosody)
out = decoder(z_tgt, spk)

assert out.shape == wav.shape
print('✅ パイプラインOK')
"
```

## 設計

2026年arXiv音声論文22本の文献レビューに基づく。  
詳細設計ドキュメントは `designs/` に格納。  
5エージェント並列レビュー + スコアリングにより最優秀アーキテクチャを選定。

## 状況

- [x] F³-Encoder (因果的ConvNeXt v2)
- [x] F³-Decoder (MRFアップサンプラ + FiLM)
- [x] VectorFieldNet (CFM, 12ブロック)
- [x] CFM Loss (OTパス + シグマ正則化)
- [x] Euler/RK4 ODEソルバ
- [x] 形状テスト合格 (91.6Mパラメータ)
- [ ] 話者エンコーダ + プロンプトトークン
- [ ] 韻律抽出器
- [ ] 学習パイプライン
- [ ] ストリーミング推論

## ライセンス

MIT
