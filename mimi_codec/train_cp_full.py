"""Full-scale CodePredictor training on MPS for cross-text VC."""
import sys; sys.path.insert(0, '/Users/asill/btrv5')
import torch, torch.nn as nn, torch.nn.functional as F, soundfile as sf
from moshi.models import loaders; from pathlib import Path
from scipy import signal
from flowvc.code_predictor import SpeakerEmbedder, CodePredictor
import subprocess, random

device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f'Device: {device}')

# Load Mimi (on CPU, only used for final decode)
mimi = loaders.get_mimi(Path('/Users/asill/.cache/huggingface/hub/models--kyutai--moshiko-pytorch-bf16/snapshots/2bfc9ae6e89079a5cc7ed2a68436010d91a3d289/tokenizer-e351c8d8-checkpoint125.safetensors'))

# Load cached codes
cache = torch.load('/Users/asill/btrv5/runs/vctk_codes_31spk.pt')
print(f'Loaded {len(cache)} cached codes')

speakers_list = sorted(set(k[0] for k in cache.keys()))
utts_list = sorted(set(k[1] for k in cache.keys()))
print(f'{len(speakers_list)} speakers, {len(utts_list)} utterances')

# Build training pairs: same text, different speakers
pairs = []
for utt in utts_list:
    spks_with = [s for s in speakers_list if (s, utt) in cache]
    if len(spks_with) < 2: continue
    for s in spks_with:
        for t in spks_with:
            if s != t:
                pairs.append((s, t, utt))

random.shuffle(pairs)
print(f'Training pairs: {len(pairs)}')

# Create model
speaker_emb = SpeakerEmbedder().to(device)
code_pred = CodePredictor().to(device)
opt = torch.optim.AdamW(
    list(speaker_emb.parameters()) + list(code_pred.parameters()),
    lr=5e-4, betas=(0.9, 0.98)
)
ce = nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler() if device.type == 'mps' else None

BATCH = 32
print(f'Training (batch={BATCH}, mixed precision)...')

for step in range(200):
    random.shuffle(pairs)
    total_loss = 0
    total_acc = 0
    n_batches = 0
    
    for b_start in range(0, min(len(pairs), 2000), BATCH):
        batch = pairs[b_start:b_start+BATCH]
        B = len(batch)
        
        # Prepare batch on device
        lv0_list, lv1_7_list = [], []
        for s, t, utt in batch:
            codes_src = cache[(s, utt)]
            codes_tgt = cache[(t, utt)]
            lv0_list.append(codes_src[0, 0])       # (T,)
            lv1_7_list.append(codes_tgt[0, 1:])     # (7, T)
        
        lv0 = torch.stack(lv0_list).to(device).long()  # (B, T)
        lv1_7_gt = torch.stack(lv1_7_list).to(device).long()  # (B, 7, T)
        
        # Forward
        with torch.amp.autocast('mps' if device.type == 'mps' else 'cpu'):
            spk = speaker_emb(lv1_7_gt)
            logits_list = code_pred(lv0, spk)
            loss = sum(ce(logits_list[i].reshape(-1, 2048), lv1_7_gt[:, i, :].reshape(-1)) for i in range(7))
        
        opt.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(list(speaker_emb.parameters()) + list(code_pred.parameters()), 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(speaker_emb.parameters()) + list(code_pred.parameters()), 1.0)
            opt.step()
        
        total_loss += loss.item() * B
        n_batches += 1
    
    if step % 20 == 0:
        avg_loss = total_loss / (n_batches * BATCH)
        # Quick eval
        with torch.no_grad():
            pred = code_pred.predict(lv0[:1], spk[:1])
            acc = (pred == lv1_7_gt[:1]).float().mean().item()
        print(f'  step {step:3d}: loss={avg_loss:.4f} | train_acc={acc:.4f}')

# Save
torch.save({
    'speaker_emb': speaker_emb.state_dict(),
    'code_pred': code_pred.state_dict(),
}, '/Users/asill/btrv5/runs/code_predictor_mps.pt')
print('Saved code_predictor_mps.pt')

# Cross-text VC test
print('\
Cross-text VC test...')
subprocess.run(['ffmpeg','-y','-i','/Users/asill/Downloads/origin.mp3','-ar','24000','-ac','1','-sample_fmt','s16','/tmp/t_24k.wav'], capture_output=True)

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
tgt_audio = load('/tmp/t_24k.wav', dur=None)
src_audio = load(f'{base}/p225/p225_001_mic1.flac')

with torch.no_grad():
    cs = mimi.encode(src_audio)
    ct = mimi.encode(tgt_audio)
    Tv = min(cs.shape[2], ct.shape[2])
    
    # Speaker embedding from target (on device)
    tgt_lv1_7 = ct[:, 1:, :Tv].to(device)
    spk_tgt = speaker_emb(tgt_lv1_7)
    
    # Predict LV1-7 for source content
    src_lv0 = cs[:, 0, :Tv].to(device)
    pred_lv1_7 = code_pred.predict(src_lv0, spk_tgt)  # (1, 7, T)
    
    # Combine and decode
    codes_vc = torch.cat([src_lv0.cpu().unsqueeze(1), pred_lv1_7.cpu()], dim=1)
    vc = mimi.decode(codes_vc)
    
    out = '/Users/asill/research5'
    sf.write(f'{out}/vc_mps.wav', vc.squeeze().numpy(), 24000)
    sf.write(f'{out}/vc_mps_src.wav', mimi.decode(cs[:,:,:Tv]).squeeze().numpy(), 24000)
    sf.write(f'{out}/vc_mps_tgt.wav', mimi.decode(ct[:,:,:Tv]).squeeze().numpy(), 24000)
    
    # Metrics
    zv = mimi.encode_to_latent(vc[:,:,:min(vc.shape[2]*1920, src_audio.shape[2])//1920], quantize=False)
    zs = mimi.encode_to_latent(src_audio[:,:,:Tv*STRIDE], quantize=False)
    zt = mimi.encode_to_latent(tgt_audio[:,:,:Tv*STRIDE], quantize=False)
    T2 = min(zv.shape[2], zs.shape[2], zt.shape[2])
    cs_cos = F.cosine_similarity(zv[:,:,:T2].reshape(-1), zs[:,:,:T2].reshape(-1), dim=0)
    ct_cos = F.cosine_similarity(zv[:,:,:T2].reshape(-1), zt[:,:,:T2].reshape(-1), dim=0)
    print(f'  cos_src={cs_cos:.4f} cos_tgt={ct_cos:.4f} Δ={ct_cos-cs_cos:+.4f}')
    print(f'  ✅ {out}/vc_mps.wav')
