"""Train Mimi Latent Converter on VCTK parallel utterances.
MSE loss: z_vc = converter(z_src, s_tgt) should match z_tgt for same-text pairs.
"""
import sys; sys.path.insert(0, '/Users/asill/btrv5')
import torch, torch.nn as nn, torch.nn.functional as F, soundfile as sf
from moshi.models import loaders; from pathlib import Path
from scipy import signal
from flowvc.mimi_latent_converter import LatentConverter, MimiVCTrainer

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
speakers = ['p225','p226','p227','p228','p229']
utts = ['001','002','003','004','005']  # 5 parallel utterances

converter = LatentConverter()
opt = torch.optim.AdamW(converter.parameters(), lr=5e-4, betas=(0.9, 0.98))

print('Training LatentConverter (parallel utterance MSE)...')
for step in range(500):
    total_loss = 0
    for utt in utts:
        for src_spk in speakers:
            for tgt_spk in speakers:
                if src_spk == tgt_spk: continue
                src = load(f'{base}/{src_spk}/{src_spk}_{utt}_mic1.flac')
                tgt = load(f'{base}/{tgt_spk}/{tgt_spk}_{utt}_mic1.flac')
                
                with torch.no_grad():
                    z_src = mimi.encode_to_latent(src, quantize=False)
                    z_tgt = mimi.encode_to_latent(tgt, quantize=False)
                T = min(z_src.shape[2], z_tgt.shape[2])
                z_src, z_tgt = z_src[:,:,:T], z_tgt[:,:,:T]
                
                s_tgt = z_tgt.mean(dim=2)  # simple speaker embedding
                z_vc = converter(z_src, s_tgt)
                
                loss = F.mse_loss(z_vc, z_tgt)
                total_loss += loss
    
    opt.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(converter.parameters(), 1.0)
    opt.step()
    
    if step % 50 == 0:
        avg_loss = total_loss.item() / (len(speakers) * (len(speakers)-1) * len(utts))
        print(f'  step {step:3d}: loss={avg_loss:.6f}')

# Test
print('\
Testing...')
with torch.no_grad():
    src = load(f'{base}/p225/p225_001_mic1.flac')
    tgt = load(f'{base}/p226/p226_001_mic1.flac')
    z_src = mimi.encode_to_latent(src, quantize=False)
    z_tgt = mimi.encode_to_latent(tgt, quantize=False)
    T = min(z_src.shape[2], z_tgt.shape[2])
    
    s_tgt = z_tgt[:,:,:T].mean(dim=2)
    z_vc = converter(z_src[:,:,:T], s_tgt)
    
    # Decode
    codes_vc = mimi.quantizer.encode(z_vc)
    vc = mimi.decode(codes_vc)
    sf.write('/Users/asill/research5/mimi_latent_v1.wav', vc.squeeze().numpy(), 24000)
    
    # Baseline token swap
    codes_src = mimi.quantizer.encode(z_src[:,:,:T])
    codes_tgt = mimi.quantizer.encode(z_tgt[:,:,:T])
    codes_base = codes_src.clone()
    codes_base[:, 1:, :] = codes_tgt[:, 1:, :]
    vc_base = mimi.decode(codes_base)
    sf.write('/Users/asill/research5/mimi_latent_base.wav', vc_base.squeeze().numpy(), 24000)
    
    # Metrics
    for name, zv in [('conv', z_vc), ('base', mimi.encode_to_latent(vc_base, quantize=False))]:
        z_vc_enc = mimi.encode_to_latent(vc if name=='conv' else vc_base, quantize=False)
        T2 = min(z_vc_enc.shape[2], z_src.shape[2], z_tgt.shape[2])
        cs = F.cosine_similarity(z_vc_enc[:,:,:T2].reshape(-1), z_src[:,:,:T2].reshape(-1), dim=0)
        ct = F.cosine_similarity(z_vc_enc[:,:,:T2].reshape(-1), z_tgt[:,:,:T2].reshape(-1), dim=0)
        print(f'  {name}: cos_src={cs:.4f} cos_tgt={ct:.4f} Δ={ct-cs:+.4f}')
    
    print(f'  ✅ /Users/asill/research5/mimi_latent_v1.wav')
    print(f'  ✅ /Users/asill/research5/mimi_latent_base.wav')
