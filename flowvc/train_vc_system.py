"""Train MimiVCSystem on cached VCTK latents and test cross-text VC."""
import sys; sys.path.insert(0, '/Users/asill/btrv5')
import torch, torch.nn.functional as F, soundfile as sf, subprocess
from moshi.models import loaders; from pathlib import Path
from scipy import signal
from flowvc.mimi_vc_system import MimiVCSystem

mimi = loaders.get_mimi(Path('/Users/asill/.cache/huggingface/hub/models--kyutai--moshiko-pytorch-bf16/snapshots/2bfc9ae6e89079a5cc7ed2a68436010d91a3d289/tokenizer-e351c8d8-checkpoint125.safetensors'))
for p in mimi.parameters(): p.requires_grad_(False)

vc = MimiVCSystem(mimi)
opt = torch.optim.AdamW(vc.parameters(), lr=5e-4, betas=(0.9, 0.98))

STRIDE = 1920

def load(path, dur=2):
    data, sr = sf.read(path)
    if sr != 24000: data = signal.resample(data, int(len(data)*24000/sr), axis=0)
    if dur is not None:
        L = dur * 24000 - (dur * 24000 % STRIDE)
        data = data[:L]
    else:
        L = len(data) - (len(data) % STRIDE)
        data = data[:L]
    if data.ndim > 1: data = data.mean(axis=1)
    return torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)

base = '/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed'

# Load cached latents (continuous z from Mimi)
z_cache = torch.load('/Users/asill/btrv5/runs/mimi_latent_cache.pt')
speakers = sorted(set(k[0] for k in z_cache.keys()))
utts = sorted(set(k[1] for k in z_cache.keys()))

# Pad all to same T
T_max = max(z.shape[2] for z in z_cache.values())
for k in z_cache:
    z = z_cache[k]
    if z.shape[2] < T_max:
        z_cache[k] = F.pad(z, (0, T_max - z.shape[2]))

print(f'Cached: {len(z_cache)} latents, {len(speakers)} spks, {len(utts)} utts, T={T_max}')

# Build training pairs: same text, different speakers
pairs = []
for utt in utts:
    spks_with = [s for s in speakers if (s, utt) in z_cache]
    for s in spks_with:
        for t in spks_with:
            if s != t:
                pairs.append((s, t, utt))

print(f'Training pairs: {len(pairs)}')

print('Training MimiVCSystem...')
for step in range(200):
    total_loss = 0
    metrics = {'mse': 0, 'code': 0, 'content': 0, 'speaker': 0}
    
    for s, t, utt in pairs:
        z_src = z_cache[(s, utt)]
        z_tgt = z_cache[(t, utt)]
        
        loss, m = vc.training_loss(z_src, z_tgt)
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vc.parameters(), 1.0)
        opt.step()
        
        total_loss += loss.item()
        for k in metrics:
            metrics[k] += m[k]
    
    n = len(pairs)
    if step % 20 == 0:
        print(f'  step {step:3d}: loss={total_loss/n:.4f} mse={metrics["mse"]/n:.4f} '
              f'code={metrics["code"]/n:.4f} content={metrics["content"]/n:.4f} spk={metrics["speaker"]/n:.4f}')

# Save
torch.save(vc.state_dict(), '/Users/asill/btrv5/runs/mimi_vc_system.pt')

# Cross-text VC test
print('\
Cross-text VC test...')
subprocess.run(['ffmpeg','-y','-i','/Users/asill/Downloads/origin.mp3','-ar','24000','-ac','1','-sample_fmt','s16','/tmp/t_24k.wav'], capture_output=True)
tgt_audio = load('/tmp/t_24k.wav', dur=None)
src_audio = load(f'{base}/p225/p225_001_mic1.flac')

with torch.no_grad():
    vc_audio = vc.convert(src_audio, tgt_audio)
    
    out = '/Users/asill/research5'
    # Trim to same length
    vc_audio = vc_audio[:, :, :src_audio.shape[2]]
    sf.write(f'{out}/vc_system.wav', vc_audio.squeeze().numpy(), 24000)
    
    z_src = mimi.encode_to_latent(src_audio, quantize=False)
    z_tgt = mimi.encode_to_latent(tgt_audio, quantize=False)
    z_vc = mimi.encode_to_latent(vc_audio, quantize=False)
    
    T2 = min(z_vc.shape[2], z_src.shape[2], z_tgt.shape[2])
    cs = F.cosine_similarity(z_vc[:,:,:T2].reshape(-1), z_src[:,:,:T2].reshape(-1), dim=0)
    ct = F.cosine_similarity(z_vc[:,:,:T2].reshape(-1), z_tgt[:,:,:T2].reshape(-1), dim=0)
    print(f'  cos_src={cs:.4f} cos_tgt={ct:.4f} Δ={ct-cs:+.4f}')
    print(f'  ✅ {out}/vc_system.wav')
