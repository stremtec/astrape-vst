#!/bin/bash
# Apply all Astrape VC improvements to vanilla train_mcs_q2d2.py
# Run from btrv5 root: bash apply_all_patches.sh

cd /Users/asill/btrv5

python3 << 'PATCHEND'
import re

with open('train_mcs_q2d2.py','r') as f: content = f.read()

# ============================================================
# 1. Add imports needed for new features
# ============================================================
old_import = 'from mcs_common import (\n    Batch, MioCompactDataset, ContentCollator,\n    split_by_speaker, speaker_balanced_subset,\n    move_batch, save_checkpoint,\n    CausalConv1d, ResidualConvBlock, CellDownsample,'
new_import = 'from mcs_common import (\n    Batch, MioCompactDataset, ContentCollator,\n    split_by_speaker, speaker_balanced_subset,\n    move_batch, save_checkpoint,\n    CausalConv1d, ResidualConvBlock, DepthwiseResidualBlock, CellDownsample,'
content = content.replace(old_import, new_import)

# ============================================================
# 2. Add config fields: stem_block_type, time_shift, forecast, contrastive, GRL num speakers, noise_dropout, l2_norm
# ============================================================
old_config = '''    conv_kernel: int = 5
    stem_dilations: tuple[int, ...] = (1, 2, 4, 8)'''
new_config = '''    conv_kernel: int = 5
    stem_dilations: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16)
    stem_block_type: str = "depthwise"  # "standard" | "depthwise"
    q2d2_noise_dropout: float = 0.0  # exploration noise for Q2D2
    q2d2_l2_norm: bool = False  # L2-normalize features before grid snapping'''
content = content.replace(old_config, new_config)

old_grl = '    grl_num_speakers: int = 0        # set automatically from dataset'
new_grl = '    grl_num_speakers: int = 0        # set automatically from dataset\n    use_wavlm_frontend: bool = False  # use WavLM CNN instead of Mel'
content = content.replace(old_grl, new_grl)

# ============================================================
# 3. Add DepthwiseResidualBlock support in encoder init
# ============================================================
old_stem = '''        # ── conv frontend (unchanged) ──
        self.input_conv = CausalConv1d(config.in_dim, dim, config.conv_kernel)
        self.blocks = nn.ModuleList([
            ResidualConvBlock(dim, config.conv_kernel, d, config.dropout)
            for d in config.stem_dilations
        ])'''
new_stem = '''        # ── conv frontend (depthwise-separable for deeper receptive field) ──
        Block = DepthwiseResidualBlock if config.stem_block_type == "depthwise" else ResidualConvBlock
        self.input_conv = CausalConv1d(config.in_dim, dim, config.conv_kernel)
        self.blocks = nn.ModuleList([
            Block(dim, config.conv_kernel, d, config.dropout)
            for d in config.stem_dilations
        ])'''
content = content.replace(old_stem, new_stem)

# ============================================================
# 4. Add Q2D2 noise_dropout and l2_norm to quantizer init
# ============================================================
old_q2d2_init = '''        self.q2d2 = Q2D2Projection(
            encoder_dim=config.trans_dim,
            q2d2_dim=config.q2d2_dim,
            content_dim=config.content_dim,
            levels=list(config.q2d2_levels),
            vq_type=config.q2d2_grid,
        )'''
new_q2d2_init = '''        self.q2d2 = Q2D2Projection(
            encoder_dim=config.trans_dim,
            q2d2_dim=config.q2d2_dim,
            content_dim=config.content_dim,
            levels=list(config.q2d2_levels),
            vq_type=config.q2d2_grid,
            noise_dropout=config.q2d2_noise_dropout,
            use_l2_norm=config.q2d2_l2_norm,
        )'''
content = content.replace(old_q2d2_init, new_q2d2_init)

# ============================================================
# 5. Add time-shift support to q2d2_losses
# ============================================================
old_loss_sig = '''def q2d2_losses(
    output: dict,
    batch: Batch,
    args: argparse.Namespace,
    quantizer: Q2D2Quantizer | None = None,
    speaker_classifier: nn.Module | None = None,
    speaker_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:'''
new_loss_sig = '''def q2d2_losses(
    output: dict,
    batch: Batch,
    args: argparse.Namespace,
    quantizer: Q2D2Quantizer | None = None,
    speaker_classifier: nn.Module | None = None,
    speaker_ids: torch.Tensor | None = None,
    time_shift: int = 0,
) -> tuple[torch.Tensor, dict[str, float]]:'''
content = content.replace(old_loss_sig, new_loss_sig)

# Add time_shift logic after projected extraction
old_proj = '''    projected = output["projected"]                     # (B, 768, T)
    q2d2_codes = output.get("q2d2_codes")               # (B, T, 6) or None

    length = min(projected.shape[2], batch.content.shape[1],
                 batch.mask.shape[1])
    mask = batch.mask[:, :length]

    pred_768 = projected[:, :, :length]                  # (B, 768, L)
    tgt_768 = batch.content[:, :length]                  # (B, L, 768)'''
new_proj = '''    projected = output["projected"]                     # (B, 768, T)
    q2d2_codes = output.get("q2d2_codes")               # (B, T, 6) or None

    ts = time_shift
    length = min(projected.shape[2] - ts, batch.content.shape[1] - ts,
                 batch.mask.shape[1] - ts)
    if length < 2:
        return projected.sum() * 0.0, {"cos768": 0.0}
    mask = batch.mask[:, ts:ts + length]

    # ── time-shifted alignment ──
    # student[t] compares with teacher[t-ts]
    pred_768 = projected[:, :, ts:ts + length]           # (B, 768, L)
    if ts > 0:
        tgt_768 = batch.content[:, :length]               # student[ts..] ↔ teacher[0..]
    else:
        tgt_768 = batch.content[:, :length]               # (B, L, 768)'''
content = content.replace(old_proj, new_proj)

# ============================================================
# 6. Add forecast heads and SSL heads to model
# ============================================================
old_q2d2_end = '''        self.q2d2 = Q2D2Projection(
            encoder_dim=config.trans_dim,
            q2d2_dim=config.q2d2_dim,
            content_dim=config.content_dim,
            levels=list(config.q2d2_levels),
            vq_type=config.q2d2_grid,
            noise_dropout=config.q2d2_noise_dropout,
            use_l2_norm=config.q2d2_l2_norm,
        )

        # ── optional GRL speaker classifier ──'''
new_q2d2_end = '''        self.q2d2 = Q2D2Projection(
            encoder_dim=config.trans_dim,
            q2d2_dim=config.q2d2_dim,
            content_dim=config.content_dim,
            levels=list(config.q2d2_levels),
            vq_type=config.q2d2_grid,
            noise_dropout=config.q2d2_noise_dropout,
            use_l2_norm=config.q2d2_l2_norm,
        )

        # ── optional WavLM frontend adapter ──
        self.wavlm_adapter = None

        # ── forecast heads: predict teacher[t+1], teacher[t+2] ──
        self.forecast_head_1 = nn.Linear(config.trans_dim, config.content_dim)
        self.forecast_head_2 = nn.Linear(config.trans_dim, config.content_dim)

        # ── optional GRL speaker classifier ──'''
content = content.replace(old_q2d2_end, new_q2d2_end)

# Add forecast outputs to forward return
old_forward_return = '''        return {
            "projected": content.transpose(1, 2),   # (B, 768, T)
            "q2d2_codes": q2d2_codes,                # (B, T, 6)
            "ordinal": None,                          # no ordinal heads in Q2D2
        }'''
new_forward_return = '''        # ── forecast predictions ──
        fc1 = self.forecast_head_1(h)  # (B, T, 768)
        fc2 = self.forecast_head_2(h)

        return {
            "projected": content.transpose(1, 2),   # (B, 768, T)
            "q2d2_codes": q2d2_codes,                # (B, T, 6)
            "ordinal": None,
            "forecast_1": fc1.transpose(1, 2),        # (B, 768, T)
            "forecast_2": fc2.transpose(1, 2),
        }'''
content = content.replace(old_forward_return, new_forward_return)

# Add forecast loss after delta loss
old_delta_end = '''    loss = (args.content_cos_weight * cos768_loss +
            args.content_l1_weight * content_l1 +
            args.delta_weight * delta)'''
new_delta_end = '''    loss = (args.content_cos_weight * cos768_loss +
            args.content_l1_weight * content_l1 +
            args.delta_weight * delta)

    # ── forecast loss ──
    forecast_weight = getattr(args, "forecast_weight", 0.0)
    forecast_loss_val: float = 0.0
    if forecast_weight > 0:
        fc1 = output.get("forecast_1")
        fc2 = output.get("forecast_2")
        if fc1 is not None and fc2 is not None and length >= 3:
            fc1_flat = fc1[:, :, ts:ts + length].permute(0, 2, 1)
            fc2_flat = fc2[:, :, ts:ts + length].permute(0, 2, 1)
            Lf = min(length, batch.content.shape[1] - 2)
            tgt_fc1 = batch.content[:, 1:1 + Lf]
            tgt_fc2 = batch.content[:, 2:2 + Lf]
            fl1 = F.mse_loss(fc1_flat[:, :Lf, :], tgt_fc1, reduction="mean")
            fl2 = F.mse_loss(fc2_flat[:, :Lf, :], tgt_fc2, reduction="mean")
            fl = (fl1 + fl2) * 0.5
            forecast_loss_val = float(fl.detach().cpu())
            loss = loss + forecast_weight * fl'''
content = content.replace(old_delta_end, new_delta_end)

# Add forecast_loss to metrics
old_metrics1 = '        "grl_loss": grl_loss_val,\n        "grl_acc": grl_acc_val,'
new_metrics1 = '        "grl_loss": grl_loss_val,\n        "grl_acc": grl_acc_val,\n        "forecast_loss": forecast_loss_val,'
content = content.replace(old_metrics1, new_metrics1)

# ============================================================
# 7. Add CLI flags
# ============================================================
old_cli = '''    p.add_argument("--grl-weight", type=float, default=0.0,
                   help="GRL speaker disentanglement weight (0=disabled, ~0.1).")'''
new_cli = '''    p.add_argument("--grl-weight", type=float, default=0.0,
                   help="GRL speaker disentanglement weight (0=disabled, ~0.1).")
    p.add_argument("--grl-num-speakers", type=int, default=0,
                   help="Number of speakers for GRL classifier (auto if 0).")
    p.add_argument("--time-shift", type=int, default=0,
                   help="Shift teacher target by Δ frames. 1 frame = 40ms.")
    p.add_argument("--forecast-weight", type=float, default=0.0,
                   help="Weight on forecast heads.")
    p.add_argument("--stem-block-type", default="depthwise",
                   choices=["standard","depthwise"],
                   help="Conv stem block type.")
    p.add_argument("--center-false", action="store_true",
                   help="Compute center=False mel on-the-fly from raw audio.")
    p.add_argument("--voiced-boost", type=float, default=1.0,
                   help="Voiced frame weight multiplier.")'''
content = content.replace(old_cli, new_cli)

# ============================================================
# 8. Pass flags to config
# ============================================================
old_cfg_pass = '''        q2d2_grid=args.q2d2_grid,
        grl_weight=args.grl_weight,
        grl_num_speakers=len(unique_speakers),
    )'''
new_cfg_pass = '''        q2d2_grid=args.q2d2_grid,
        grl_weight=args.grl_weight,
        grl_num_speakers=args.grl_num_speakers if args.grl_num_speakers > 0 else len(unique_speakers),
        stem_block_type=args.stem_block_type,
    )'''
content = content.replace(old_cfg_pass, new_cfg_pass)

# ============================================================
# 9. Add center=False wrapper hook in training loop
# ============================================================
old_loader = '''    train_loader = DataLoader('''
new_loader = '''    if args.center_false:
        from eval_mcs_trans_audio import SAMPLE_RATE
        train_ds = CenterFalseMelWrapper(train_ds, source_files)
        probe_ds = CenterFalseMelWrapper(probe_ds, source_files)
        print("center=False mel: computing on-the-fly from raw audio", flush=True)

    train_loader = DataLoader('''
content = content.replace(old_loader, new_loader)

# ============================================================
# 10. Fix resume to use strict=False
# ============================================================
old_resume = 'model.load_state_dict(checkpoint["state_dict"], strict=True)'
new_resume = 'missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)\n        if missing:\n            print(f"Missing keys: {len(missing)}\", flush=True)'
content = content.replace(old_resume, new_resume)

# ============================================================
# 11. Add CenterFalseMelWrapper class
# ============================================================
center_false_class = '''

# ── Center=False Mel Wrapper ──
class CenterFalseMelWrapper(Dataset):
    """Wraps dataset, replacing cached center=True mel with on-the-fly center=False."""
    def __init__(self, base_dataset, source_files):
        self.base = base_dataset; self.src = source_files
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        sample = self.base[idx]; si = int(sample['idx'])
        import soundfile as sf
        w, sr = sf.read(str(Path(self.src[si])), dtype='float32')
        w = torch.from_numpy(np.asarray(w))
        if w.ndim == 2: w = w.mean(1)
        if sr != SAMPLE_RATE:
            w = torchaudio.functional.resample(w.unsqueeze(0), sr, SAMPLE_RATE).squeeze(0)
        mel = torchaudio.transforms.MelSpectrogram(
            SAMPLE_RATE, 2048, 882, n_mels=80, f_min=0.0, f_max=SAMPLE_RATE/2.0,
            power=1, center=False
        )(w.unsqueeze(0))
        mel = torch.log(torch.clamp(mel, min=1e-5))
        sample['mel'] = mel[0]
        return sample
'''

# Insert before ContentCollator
old_collator = 'class ContentCollator:'
content = content.replace(old_collator, center_false_class + '\n' + old_collator)

# ============================================================
# 12. Patch evaluate and train loop to pass time_shift
# ============================================================
old_eval = '''        _, metrics = q2d2_losses(output, batch, args, quantizer,
                                 model.speaker_classifier, speaker_ids)'''
new_eval = '''        _, metrics = q2d2_losses(output, batch, args, quantizer,
                                 model.speaker_classifier, speaker_ids,
                                 time_shift=args.time_shift)'''
content = content.replace(old_eval, new_eval)

old_train = '''            loss, metrics = q2d2_losses(output, batch, args, quantizer,
                                        model.speaker_classifier, speaker_ids)'''
new_train = '''            loss, metrics = q2d2_losses(output, batch, args, quantizer,
                                        model.speaker_classifier, speaker_ids,
                                        time_shift=args.time_shift)'''
content = content.replace(old_train, new_train)

# Add optimizer mismatch handling
old_opt_load = '        if "optimizer" in checkpoint:\n            optimizer.load_state_dict(checkpoint["optimizer"])'
new_opt_load = '        if "optimizer" in checkpoint:\n            try:\n                optimizer.load_state_dict(checkpoint["optimizer"])\n            except ValueError:\n                print("Optimizer mismatch, starting fresh")'
content = content.replace(old_opt_load, new_opt_load)

with open('train_mcs_q2d2.py','w') as f: f.write(content)
print(f'Patched. Lines: {len(content.splitlines())}')
PATCHEND

# Verify syntax
.venv/bin/python3 -c "compile(open('train_mcs_q2d2.py').read(),'train_mcs_q2d2.py','exec');print('Syntax OK')"
echo "Done!"
