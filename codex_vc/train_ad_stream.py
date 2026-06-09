"""Train AudioDec StreamVC with differentiable decoder."""
import sys, os, random; sys.path.insert(0, '.'); sys.path.insert(0, '/Users/asill/btrvrc0')
import torch, torch.nn as nn, torch.nn.functional as F, soundfile as sf, time, subprocess
from transformers import HubertModel
from v3lite.codec_audiodec import AudioDecCodec
from scipy import signal
import numpy as np
from codex_vc.audiodec_stream import AudioDecStreamVC
from codex_vc.metrics import compute_all_metrics, format_metrics

SR_AUDIODEC = 48000
HOP = 300
STEPS = 100
LR = 5e-4

print("Loading models...")
hubert = HubertModel.from_pretrained('facebook/hubert-base-ls960').eval()
codec = AudioDecCodec(device='cpu')
for p in codec.parameters(): p.requires_grad_(False)

base = '/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed'
spks = ['p225','p226','p227','p228','p229','p230','p231','p232','p233','p234','p235','p236']
utts = ['001','002','003']

spk_to_idx = {s: i for i, s in enumerate(spks)}
n_spk = len(spks)
print(f"  {n_spk} speakers")

# Pre-compute HuBERT features + AudioDec targets
print("Pre-computing...")
hubert_feats = {}
ad_targets = {}

for s in spks:
    for u in utts:
        f = f'{base}/{s}/{s}_{u}_mic1.flac'
        if not os.path.isfile(f): continue
        
        # Load 16kHz for HuBERT
        d, sr = sf.read(f)
        if sr != 16000: d = signal.resample(d, int(len(d)*16000/sr), axis=0)
        if d.ndim > 1: d = d.mean(axis=1)
        d16 = d[:16000*2]
        src_16k = torch.from_numpy(d16).float().unsqueeze(0)
        
        with torch.no_grad():
            hs = hubert(src_16k, output_hidden_states=True).hidden_states
            h_avg = (hs[1] + hs[2] + hs[3]) / 3.0
        
        # Load 48kHz for AudioDec target
        d48, sr48 = sf.read(f)
        if sr48 != SR_AUDIODEC:
            d48 = signal.resample(d48, int(len(d48)*SR_AUDIODEC/sr48), axis=0)
        if d48.ndim > 1: d48 = d48.mean(axis=1)
        L = int(1.5 * SR_AUDIODEC) - (int(1.5 * SR_AUDIODEC) % HOP)
        d48 = d48[:L]
        tgt_audio = torch.from_numpy(d48).float()
        
        # AudioDec encode target
        with torch.no_grad():
            z_tgt = codec.encode(tgt_audio)  # (T_ad, 64)
        
        hubert_feats[(s, u)] = h_avg
        ad_targets[(s, u)] = (z_tgt, tgt_audio)

T_h = min(h.shape[1] for h in hubert_feats.values())
hubert_feats = {k: v[:, :T_h] for k, v in hubert_feats.items()}
print(f"  {len(hubert_feats)} features, T_h={T_h}")

# Model
model = AudioDecStreamVC(hubert, codec, n_speakers=n_spk)
spk_emb = nn.Embedding(n_spk, 256)
opt = torch.optim.AdamW(list(model.parameters()) + list(spk_emb.parameters()), lr=LR)
ce_spk = nn.CrossEntropyLoss()

pairs = [(s, t, u) for u in utts for s in spks for t in spks
         if s != t and (s,u) in hubert_feats and (t,u) in hubert_feats]
random.shuffle(pairs)
n_train = int(len(pairs) * 0.8)
train_p = pairs[:n_train]; val_p = pairs[n_train:]
print(f"  Train pairs: {len(train_p)}, Val pairs: {len(val_p)}")

print()
print(f"Training {STEPS} steps...")
t0 = time.time()
best_val = float('inf')

for step in range(STEPS):
    random.shuffle(train_p)
    tl = 0; tl_r = 0; tl_a = 0
    max_p = min(len(train_p), 20)  # AudioDec decode is slow
    
    for s, t, u in train_p[:max_p]:
        h_avg = hubert_feats[(s, u)]
        z_tgt, tgt_audio = ad_targets[(t, u)]
        tgt_spk = spk_emb(torch.tensor([spk_to_idx[t]]))
        
        # Forward
        z_vc = model.forward(
            torch.zeros(1, 16000), tgt_spk  # dummy audio, use pre-computed h_avg
        )
        # Manual forward with pre-computed h_avg
        with torch.no_grad():
            h_avg_b = h_avg  # already (1, T_h, 768)
        
        # Adversarial
        spk_logits = model.spk_adversarial(h_avg_b.transpose(1, 2))
        loss_adv = ce_spk(spk_logits, torch.tensor([spk_to_idx[s]]))
        
        # Content projection
        z_content = model.content_proj(h_avg_b)
        gamma = model.spk_gamma(tgt_spk).unsqueeze(1)
        beta = model.spk_beta(tgt_spk).unsqueeze(1)
        mean = z_content.mean(dim=1, keepdim=True)
        std = z_content.std(dim=1, keepdim=True) + 1e-5
        z_vc = (z_content - mean) / std * gamma + beta
        
        # Decode through AudioDec (differentiable!)
        z_vc_2d = z_vc.squeeze(0)  # (T, 64)
        audio_vc = codec.decode(z_vc_2d)  # (T_samples,) TENSOR
        
        T_a = min(len(audio_vc), len(tgt_audio))
        loss_rec = F.mse_loss(audio_vc[:T_a], tgt_audio[:T_a])
        
        loss = loss_rec + 1.0 * loss_adv
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tl += loss.item(); tl_r += loss_rec.item(); tl_a += loss_adv.item()
    
    if step % 30 == 0 or step == STEPS - 1:
        print(f"  step {step:4d}: rec={tl_r/max_p:.4f} adv={tl_a/max_p:.4f} "
              f"[{time.time()-t0:.0f}s]")

torch.save({'model': model.state_dict(), 'spk_emb': spk_emb.state_dict()}, 'runs/ad_stream.pt')
print(f"Saved [{time.time()-t0:.0f}s]")

# Test
subprocess.run(['ffmpeg', '-y', '-i', '/Users/asill/Downloads/origin.mp3',
                '-ar', '16000', '-ac', '1', '-sample_fmt', 's16', '-t', '2',
                '/tmp/ads_test.wav'], capture_output=True)

d_src, sr_src = sf.read(f'{base}/p255/p255_001_mic1.flac')
if sr_src != 16000: d_src = signal.resample(d_src, int(len(d_src)*16000/sr_src), axis=0)
if d_src.ndim > 1: d_src = d_src.mean(axis=1)
src_16k = torch.from_numpy(d_src[:16000*2]).float().unsqueeze(0)

with torch.no_grad():
    tgt_spk = spk_emb(torch.tensor([1]))
    vc_audio = model.convert(src_16k, tgt_spk)
    
    print(f"p255->p226 (AudioDec StreamVC):")
    vc_np = vc_audio[:len(d_src[:16000*2])]
    if isinstance(vc_np, torch.Tensor): vc_np = vc_np.numpy()
    m = compute_all_metrics(d_src[:16000*2], np.zeros_like(d_src[:16000*2]), vc_np, 16000)
    print(format_metrics(m))
    sf.write('/Users/asill/research5/ad_stream.wav', vc_np, 16000)
    print("Done")
