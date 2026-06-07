"""Quick test: MimiRVQConverter on small VCTK subset."""
import sys; sys.path.insert(0, '/Users/asill/btrv5')
import torch, torch.nn as nn, torch.nn.functional as F, soundfile as sf
from moshi.models import loaders; from pathlib import Path
from scipy import signal
from flowvc.mimi_converter import MimiRVQConverter

mimi = loaders.get_mimi(Path('/Users/asill/.cache/huggingface/hub/models--kyutai--moshiko-pytorch-bf16/snapshots/2bfc9ae6e89079a5cc7ed2a68436010d91a3d289/tokenizer-e351c8d8-checkpoint125.safetensors'))
for p in mimi.parameters(): p.requires_grad_(False)

STRIDE = 1920

def load(path, dur=2):
    data, sr = sf.read(path)
    if sr != 24000: data = signal.resample(data, int(len(data)*24000/sr), axis=0)
    L = dur * 24000 - (dur * 24000 % STRIDE)
    data = data[:L]
    if data.ndim > 1: data = data.mean(axis=1)
    return torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)

base = '/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed'

# Get a batch of parallel utterances (same text, different speakers)
speakers = ['p225', 'p226', 'p227', 'p228', 'p229']
utt = '001'

with torch.no_grad():
    # Encode all samples
    codes_all = {}
    for spk in speakers:
        x = load(f'{base}/{spk}/{spk}_{utt}_mic1.flac')
        codes_all[spk] = mimi.encode(x)  # (1, 8, T)

    # Check shapes
    T = min(c.shape[2] for c in codes_all.values())
    print(f'Time steps: {T}')

# Build converter
model = MimiRVQConverter(vocab_size=2048, embed_dim=256, d_model=512, nhead=8, num_layers=4)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.98))
ce = nn.CrossEntropyLoss()

print('Training MimiRVQConverter...')
for step in range(200):
    total_loss = 0
    for src_spk in speakers:
        for tgt_spk in speakers:
            if src_spk == tgt_spk: continue
            src_lv0 = codes_all[src_spk][:, 0, :T]  # (1, T)
            tgt_lv1_7 = codes_all[tgt_spk][:, 1:, :T]  # (1, 7, T)
            tgt_lv1_7_gt = tgt_lv1_7.clone()  # ground truth

            logits = model(src_lv0, tgt_lv1_7)  # (1, 7, T, 2048)
            loss = ce(logits.reshape(-1, 2048), tgt_lv1_7_gt.reshape(-1))
            total_loss += loss

    opt.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    if step % 20 == 0:
        print(f'  step {step:3d}: loss={total_loss.item():.4f}')

# Test: predict improved codes
print('\
Testing...')
with torch.no_grad():
    src_lv0 = codes_all['p225'][:, 0, :T]
    tgt_lv1_7 = codes_all['p226'][:, 1:, :T]
    tgt_lv1_7_gt = tgt_lv1_7.clone()

    pred_codes = model.predict(src_lv0, tgt_lv1_7)  # (1, 7, T)
    
    # Accuracy
    acc = (pred_codes == tgt_lv1_7_gt).float().mean()
    print(f'  Prediction accuracy: {acc.item():.4f}')

    # Build full codes and decode
    codes_vc = torch.cat([src_lv0.unsqueeze(1), pred_codes], dim=1)  # (1, 8, T)
    vc = mimi.decode(codes_vc)
    sf.write('/Users/asill/research5/mimi_conv_v1.wav', vc.squeeze().numpy(), 24000)
    print(f'  ✅ /Users/asill/research5/mimi_conv_v1.wav')

    # Compare with baseline token swap
    codes_baseline = torch.cat([src_lv0.unsqueeze(1), tgt_lv1_7_gt], dim=1)
    vc_base = mimi.decode(codes_baseline)
    sf.write('/Users/asill/research5/mimi_conv_baseline.wav', vc_base.squeeze().numpy(), 24000)

    # Cosine comparison
    x_src = load(f'{base}/p225/p225_{utt}_mic1.flac')
    x_tgt = load(f'{base}/p226/p226_{utt}_mic1.flac')
    z_vc = mimi.encode_to_latent(vc, quantize=False)
    z_base = mimi.encode_to_latent(vc_base, quantize=False)
    z_src = mimi.encode_to_latent(x_src[:,:,:T*STRIDE], quantize=False)
    z_tgt = mimi.encode_to_latent(x_tgt[:,:,:T*STRIDE], quantize=False)
    
    T2 = min(z_vc.shape[2], z_src.shape[2], z_tgt.shape[2])
    for name, zv in [('conv', z_vc), ('base', z_base)]:
        cs = F.cosine_similarity(zv[:,:,:T2].reshape(-1), z_src[:,:,:T2].reshape(-1), dim=0)
        ct = F.cosine_similarity(zv[:,:,:T2].reshape(-1), z_tgt[:,:,:T2].reshape(-1), dim=0)
        print(f'  {name}: cos_src={cs:.4f} cos_tgt={ct:.4f} Δ={ct-cs:+.4f}')
