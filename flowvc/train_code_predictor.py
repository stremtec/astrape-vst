"""Train CodePredictor on VCTK parallel utterances for cross-text VC."""
import sys; sys.path.insert(0, '/Users/asill/btrv5')
import torch, torch.nn as nn, torch.nn.functional as F, soundfile as sf
from moshi.models import loaders; from pathlib import Path
from scipy import signal
from flowvc.code_predictor import SpeakerEmbedder, CodePredictor
import random

mimi = loaders.get_mimi(Path('/Users/asill/.cache/huggingface/hub/models--kyutai--moshiko-pytorch-bf16/snapshots/2bfc9ae6e89079a5cc7ed2a68436010d91a3d289/tokenizer-e351c8d8-checkpoint125.safetensors'))
for p in mimi.parameters(): p.requires_grad_(False)

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
utts = [f'{i:03d}' for i in range(1, 21)]  # 20 utterances

import os

# Phase 1: Cache codes for many speakers (skip if exists)
cache_path = '/Users/asill/btrv5/runs/vctk_codes_31spk.pt'
if os.path.isfile(cache_path):
    print(f'Phase 1: Loading existing cache {cache_path}')
    cache = torch.load(cache_path)
else:
    print('Phase 1: Caching codes...')
    speakers = [f'p{i}' for i in range(225, 256)]  # 31 speakers

    cache = {}
    cached_count = 0
    for spk in speakers:
        spk_dir = f'{base}/{spk}'
        if not os.path.isdir(spk_dir): continue
        for utt in utts:
            fpath = f'{spk_dir}/{spk}_{utt}_mic1.flac'
            if not os.path.isfile(fpath): continue
            try:
                x = load(fpath)
                with torch.no_grad():
                    codes = mimi.encode(x)
                cache[(spk, utt)] = codes.cpu()
                cached_count += 1
            except: pass

    # Trim to same T
    T = min(c.shape[2] for c in cache.values())
    cache = {k: v[:,:,:T] for k, v in cache.items()}
    print(f'Cached {len(cache)} entries, T={T}')
    torch.save(cache, cache_path)

# Phase 2: Train CodePredictor
print('\
Phase 2: Training CodePredictor...')

speaker_emb = SpeakerEmbedder()
code_pred = CodePredictor()
opt = torch.optim.AdamW(
    list(speaker_emb.parameters()) + list(code_pred.parameters()),
    lr=5e-4, betas=(0.9, 0.98)
)
ce = nn.CrossEntropyLoss()

# Build training pairs: same text, different speakers
pairs = []
speakers_list = list(set(k[0] for k in cache.keys()))
for utt in utts:
    spks_with_utt = [spk for spk in speakers_list if (spk, utt) in cache]
    if len(spks_with_utt) < 2: continue
    for src_spk in spks_with_utt:
        for tgt_spk in spks_with_utt:
            if src_spk != tgt_spk:
                pairs.append((src_spk, tgt_spk, utt))

random.shuffle(pairs)
print(f'Training pairs: {len(pairs)}')
if len(pairs) > 5000:
    pairs = pairs[:5000]
    print(f'  (limited to 5000)')

batch_size = 16
for step in range(300):
    total_loss = 0
    random.shuffle(pairs)
    
    for batch_start in range(0, len(pairs), batch_size):
        batch = pairs[batch_start:batch_start+batch_size]
        B = len(batch)
        
        # Prepare batch
        lv0_batch = []
        lv1_7_batch = []
        spk_emb_batch = []
        
        for src_spk, tgt_spk, utt in batch:
            codes_src = cache[(src_spk, utt)][0]  # (8, T)
            codes_tgt = cache[(tgt_spk, utt)][0]
            lv0_batch.append(codes_src[0])         # (T,)
            lv1_7_batch.append(codes_tgt[1:])      # (7, T)
        
        lv0 = torch.stack(lv0_batch)  # (B, T)
        lv1_7_gt = torch.stack(lv1_7_batch)  # (B, 7, T)
        
        # Speaker embedding from target
        spk_emb = speaker_emb(lv1_7_gt)  # (B, 512)
        
        # Predict LV1-7 codes
        logits_list = code_pred(lv0, spk_emb)  # list of (B, T, 2048)
        
        # Cross-entropy loss per level
        loss = 0
        for lv in range(7):
            loss += ce(logits_list[lv].reshape(-1, 2048), lv1_7_gt[:, lv, :].reshape(-1))
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(speaker_emb.parameters()) + list(code_pred.parameters()), 1.0)
        opt.step()
        
        total_loss += loss.item() * B
    
    avg_loss = total_loss / len(pairs)
    if step % 30 == 0:
        # Eval accuracy
        with torch.no_grad():
            pred_codes = code_pred.predict(lv0[:1], spk_emb[:1])
            acc = (pred_codes == lv1_7_gt[:1]).float().mean().item()
        print(f'  step {step:3d}: loss={avg_loss:.4f} acc={acc:.4f}')

# Save
torch.save({
    'speaker_emb': speaker_emb.state_dict(),
    'code_pred': code_pred.state_dict(),
}, '/Users/asill/btrv5/runs/code_predictor.pt')
print('Saved code_predictor.pt')

# Phase 3: Cross-text VC test
print('\
Phase 3: Cross-text VC with origin.mp3...')

# Load target audio and get speaker embedding
import subprocess
target_wav = '/tmp/target_24k.wav'
subprocess.run(['ffmpeg', '-y', '-i', '/Users/asill/Downloads/origin.mp3', 
                '-ar', '24000', '-ac', '1', '-sample_fmt', 's16', target_wav], 
               capture_output=True)

tgt = load(target_wav, dur=None)  # full
src = load(f'{base}/p225/p225_001_mic1.flac')

with torch.no_grad():
    codes_src = mimi.encode(src)
    codes_tgt = mimi.encode(tgt)
    T_vc = min(codes_src.shape[2], codes_tgt.shape[2])
    
    # Get target speaker embedding
    tgt_lv1_7 = codes_tgt[:, 1:, :T_vc]
    spk_tgt = speaker_emb(tgt_lv1_7)  # (1, 512)
    
    # Predict compatible LV1-7 for source content
    src_lv0 = codes_src[:, 0, :T_vc]
    pred_lv1_7 = code_pred.predict(src_lv0, spk_tgt)  # (1, 7, T)
    
    # Combine and decode
    codes_vc = torch.cat([src_lv0.unsqueeze(1), pred_lv1_7], dim=1)
    vc = mimi.decode(codes_vc)
    
    # Reference
    src_out = mimi.decode(codes_src[:, :, :T_vc])
    tgt_out = mimi.decode(codes_tgt[:, :, :T_vc])
    
    out = '/Users/asill/research5'
    sf.write(f'{out}/vc_cross.wav', vc.squeeze().numpy(), 24000)
    sf.write(f'{out}/vc_cross_src.wav', src_out.squeeze().numpy(), 24000)
    sf.write(f'{out}/vc_cross_tgt.wav', tgt_out.squeeze().numpy(), 24000)
    
    # Metrics
    z_vc = mimi.encode_to_latent(vc, quantize=False)
    z_src = mimi.encode_to_latent(src[:,:,:T_vc*STRIDE], quantize=False)
    z_tgt = mimi.encode_to_latent(tgt[:,:,:T_vc*STRIDE], quantize=False)
    T2 = min(z_vc.shape[2], z_src.shape[2], z_tgt.shape[2])
    
    cs = F.cosine_similarity(z_vc[:,:,:T2].reshape(-1), z_src[:,:,:T2].reshape(-1), dim=0)
    ct = F.cosine_similarity(z_vc[:,:,:T2].reshape(-1), z_tgt[:,:,:T2].reshape(-1), dim=0)
    
    print(f'  cos_src={cs:.4f} cos_tgt={ct:.4f} Δ={ct-cs:+.4f}')
    print(f'  ✅ {out}/vc_cross.wav')
