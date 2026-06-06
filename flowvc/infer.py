"""
FlowVC ストリーミング推論。

完全因果的パイプラインによるリアルタイム音声変換。
チャンク単位処理 + リングバッファ + 話者キャッシュ。

遅延内訳:
  エンコーダ:   ~15ms (6段 ConvNeXt v2)
  韻律抽出:     ~2ms  (軽量 ConvNet)
  CFM ODE:      ~12ms (4-step Euler, 12ブロック)
  デコーダ:     ~20ms (6段 MRF アップサンプラ)
  ────────────────────────────
  合計:         ~49ms / チャンク (GPU)
  TTFB:         ~49ms (最初のチャンク、話者埋め込み事前計算済み)
"""

from __future__ import annotations

import time
import torch
import torch.nn.functional as F

from .encoder import F3Encoder
from .decoder import F3Decoder
from .converter import VectorFieldNet, solve_cfm_euler
from .speaker import SpeakerEncoder
from .prosody import ProsodyExtractor


class FlowVCInference:
    """
    FlowVC ストリーミング推論エンジン。

    使用法:
        engine = FlowVCInference(encoder, decoder, vfn, speaker_enc, prosody_ext)
        engine.set_target_speaker(ref_audio)  # 事前計算

        for chunk in audio_stream:
            output_chunk = engine.process_chunk(chunk)
    """

    def __init__(
        self,
        encoder: F3Encoder,
        decoder: F3Decoder,
        vfn: VectorFieldNet,
        speaker_enc: SpeakerEncoder,
        prosody_ext: ProsodyExtractor,
        device: str = "cpu",
        chunk_ms: int = 80,        # チャンク長 (ms)
        left_ctx_ms: int = 320,    # 左コンテキスト (ms)
        ode_steps: int = 4,        # CFM ODEステップ数
    ):
        self.device = torch.device(device)
        self.encoder = encoder.to(self.device).eval()
        self.decoder = decoder.to(self.device).eval()
        self.vfn = vfn.to(self.device).eval()
        self.speaker_enc = speaker_enc.to(self.device).eval()
        self.prosody_ext = prosody_ext.to(self.device).eval()

        # タイミング
        sr = 44100
        self.chunk_samples = int(sr * chunk_ms / 1000)
        self.left_ctx_samples = int(sr * left_ctx_ms / 1000)
        self.hop_samples = 1764  # 25Hz @ 44.1kHz
        self.chunk_frames = self.chunk_samples // self.hop_samples

        self.ode_steps = ode_steps

        # 話者状態
        self.speaker_emb: torch.Tensor | None = None
        self.prompt_tokens: torch.Tensor | None = None

        # リングバッファ
        self.buffer: torch.Tensor | None = None

        # 統計
        self.stats = {"total_chunks": 0, "total_time_ms": 0.0}

    def set_target_speaker(self, ref_audio: torch.Tensor):
        """
        ターゲット話者を設定（事前計算）。

        Args:
            ref_audio: (1, T) または (B, 1, T) 参照音声 @ 44.1kHz
        """
        if ref_audio.dim() == 2:
            ref_audio = ref_audio.unsqueeze(0)  # (1, 1, T)

        ref_audio = ref_audio.to(self.device)

        with torch.no_grad():
            self.speaker_emb, self.prompt_tokens = self.speaker_enc(ref_audio)

        print(f"  話者設定完了: emb={self.speaker_emb.shape}, prompt={self.prompt_tokens.shape}")

    def process_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        """
        単一チャンクを処理。

        Args:
            chunk: (chunk_samples,) 入力音声チャンク @ 44.1kHz
        Returns:
            (chunk_samples,) 変換済み出力チャンク
        """
        if self.speaker_emb is None:
            raise RuntimeError("set_target_speaker() を先に呼んでください")

        t_start = time.time()

        # バッチ次元追加
        chunk = chunk.to(self.device).unsqueeze(0).unsqueeze(0)  # (1, 1, T_chunk)

        with torch.no_grad():
            # 1. エンコード
            z_src = self.encoder.encode(chunk)  # (1, T_lat, 768)

            # 2. 韻律抽出
            prosody = self.prosody_ext(chunk)  # (1, T_lat, 3)
            if prosody.size(1) != z_src.size(1):
                n = min(z_src.size(1), prosody.size(1))
                z_src = z_src[:, :n, :]
                prosody = prosody[:, :n, :]

            # 3. CFM ODE
            z_tgt = solve_cfm_euler(
                self.vfn, z_src, self.speaker_emb,
                self.prompt_tokens, prosody,
                n_steps=self.ode_steps,
            )

            # 4. デコード
            out = self.decoder(z_tgt, self.speaker_emb)  # (1, 1, T_out)

        elapsed = (time.time() - t_start) * 1000  # ms
        self.stats["total_chunks"] += 1
        self.stats["total_time_ms"] += elapsed

        return out.squeeze(0).squeeze(0)  # (T_out,)

    def process_stream(self, audio: torch.Tensor, chunk_ms: int | None = None) -> torch.Tensor:
        """
        長い音声をチャンク単位でストリーミング処理。

        Args:
            audio: (T,) 入力音声 @ 44.1kHz
            chunk_ms: チャンク長 (ms), Noneならデフォルト
        Returns:
            (T_out,) 変換済み出力
        """
        chunk_samples = self.chunk_samples
        if chunk_ms is not None:
            chunk_samples = int(44100 * chunk_ms / 1000)

        total_samples = audio.shape[-1]
        outputs = []

        for start in range(0, total_samples, chunk_samples):
            end = min(start + chunk_samples, total_samples)
            chunk = audio[start:end]

            # パディング（最後のチャンク）
            if chunk.shape[-1] < chunk_samples:
                chunk = F.pad(chunk, (0, chunk_samples - chunk.shape[-1]))

            out = self.process_chunk(chunk)
            outputs.append(out)

        result = torch.cat(outputs, dim=-1)
        return result[:total_samples]  # 元の長さにトリム

    @property
    def avg_latency_ms(self) -> float:
        """平均チャンク処理遅延 (ms)。"""
        if self.stats["total_chunks"] == 0:
            return 0.0
        return self.stats["total_time_ms"] / self.stats["total_chunks"]

    @property
    def rtf(self) -> float:
        """Real-Time Factor (処理時間 / 音声長)。"""
        if self.stats["total_chunks"] == 0:
            return 0.0
        audio_duration_ms = (
            self.stats["total_chunks"] * self.chunk_samples / 44100 * 1000
        )
        return self.stats["total_time_ms"] / audio_duration_ms

    def print_stats(self):
        """統計を表示。"""
        print(f"  チャンク数: {self.stats['total_chunks']}")
        print(f"  平均遅延:   {self.avg_latency_ms:.1f} ms")
        print(f"  RTF:        {self.rtf:.3f}")


# ── ファイルベース推論 ──────────────────────────────────────────

def convert_file(
    encoder: F3Encoder,
    decoder: F3Decoder,
    vfn: VectorFieldNet,
    speaker_enc: SpeakerEncoder,
    prosody_ext: ProsodyExtractor,
    src_path: str,
    ref_path: str,
    out_path: str,
    device: str = "cpu",
    ode_steps: int = 4,
):
    """
    ファイル単位の音声変換。

    Args:
        src_path: ソース音声パス
        ref_path: 参照（ターゲット話者）音声パス
        out_path: 出力パス
    """
    import torchaudio

    # 音声読み込み
    src_wav, src_sr = torchaudio.load(src_path)
    ref_wav, ref_sr = torchaudio.load(ref_path)

    # リサンプリング（必要なら）
    if src_sr != 44100:
        src_wav = torchaudio.functional.resample(src_wav, src_sr, 44100)
    if ref_sr != 44100:
        ref_wav = torchaudio.functional.resample(ref_wav, ref_sr, 44100)

    # モノラル化
    if src_wav.shape[0] > 1:
        src_wav = src_wav.mean(dim=0, keepdim=True)
    if ref_wav.shape[0] > 1:
        ref_wav = ref_wav.mean(dim=0, keepdim=True)

    engine = FlowVCInference(
        encoder, decoder, vfn, speaker_enc, prosody_ext,
        device=device, ode_steps=ode_steps,
    )

    engine.set_target_speaker(ref_wav)
    out_wav = engine.process_stream(src_wav.squeeze(0))

    # 保存
    out_wav = out_wav.unsqueeze(0).cpu()  # (1, T)
    torchaudio.save(out_path, out_wav, 44100)

    engine.print_stats()
    print(f"  保存: {out_path}")


# ── 軽量プロファイラ ────────────────────────────────────────────

def profile_pipeline(
    device: str = "cpu",
    chunk_ms: int = 80,
    ode_steps: int = 4,
):
    """
    各コンポーネントのレイテンシを計測。

    使用方法:
        python -m flowvc.infer --profile
    """
    from .encoder import make_encoder
    from .decoder import F3Decoder
    from .converter import make_vector_field_net
    from .speaker import make_speaker_encoder
    from .prosody import make_prosody_extractor
    from .config import DecoderConfig

    dev = torch.device(device)
    chunk_samples = int(44100 * chunk_ms / 1000)

    encoder = make_encoder().to(dev).eval()
    decoder = F3Decoder(DecoderConfig()).to(dev).eval()
    vfn = make_vector_field_net().to(dev).eval()
    speaker_enc = make_speaker_encoder().to(dev).eval()
    prosody_ext = make_prosody_extractor().to(dev).eval()

    wav = torch.randn(1, 1, chunk_samples, device=dev)
    ref = torch.randn(1, 1, 44100, device=dev)

    with torch.no_grad():
        spk_emb, prompt = speaker_enc(ref)

    # ウォームアップ
    for _ in range(3):
        with torch.no_grad():
            z = encoder.encode(wav)
            p = prosody_ext(wav)
            _ = solve_cfm_euler(vfn, z, spk_emb, prompt, p, n_steps=ode_steps)
            _ = decoder(z, spk_emb)

    # 計測
    times = {}

    t0 = time.time()
    for _ in range(100):
        z = encoder.encode(wav)
    times["encoder"] = (time.time() - t0) / 100 * 1000

    t0 = time.time()
    for _ in range(100):
        p = prosody_ext(wav)
    times["prosody"] = (time.time() - t0) / 100 * 1000

    t0 = time.time()
    for _ in range(50):
        z_tgt = solve_cfm_euler(vfn, z, spk_emb, prompt, p, n_steps=ode_steps)
    times["cfm_ode"] = (time.time() - t0) / 50 * 1000

    t0 = time.time()
    for _ in range(100):
        out = decoder(z, spk_emb)
    times["decoder"] = (time.time() - t0) / 100 * 1000

    total = sum(times.values())

    print(f"\n  === FlowVC プロファイル ({device}, {chunk_ms}ms chunk, {ode_steps}-step ODE) ===")
    for name, t in times.items():
        pct = t / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {name:12s} {t:6.1f} ms  ({pct:4.1f}%)  {bar}")
    print(f"  {'─'*45}")
    print(f"  {'TOTAL':12s} {total:6.1f} ms  (100.0%)")

    rtf = total / chunk_ms
    print(f"\n  RTF: {rtf:.3f}  (1.0未満でリアルタイム)")


# ── CLI ─────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="FlowVC 推論")
    parser.add_argument("--profile", action="store_true",
                        help="パイプラインプロファイル")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--chunk-ms", type=int, default=80)
    parser.add_argument("--ode-steps", type=int, default=4)
    parser.add_argument("--src", type=str, default="",
                        help="ソース音声パス（ファイル変換時）")
    parser.add_argument("--ref", type=str, default="",
                        help="参照音声パス")
    parser.add_argument("--out", type=str, default="output.wav",
                        help="出力パス")
    args = parser.parse_args()

    if args.profile:
        profile_pipeline(args.device, args.chunk_ms, args.ode_steps)

    elif args.src and args.ref:
        from .encoder import make_encoder
        from .decoder import F3Decoder
        from .converter import make_vector_field_net
        from .speaker import make_speaker_encoder
        from .prosody import make_prosody_extractor
        from .config import DecoderConfig

        device = args.device
        encoder = make_encoder().to(device).eval()
        decoder = F3Decoder(DecoderConfig()).to(device).eval()
        vfn = make_vector_field_net().to(device).eval()
        speaker_enc = make_speaker_encoder().to(device).eval()
        prosody_ext = make_prosody_extractor().to(device).eval()

        convert_file(
            encoder, decoder, vfn, speaker_enc, prosody_ext,
            src_path=args.src, ref_path=args.ref, out_path=args.out,
            device=device, ode_steps=args.ode_steps,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
