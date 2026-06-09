"""LongCat Token Swap VC Test."""
import sys, os; sys.path.insert(0, '/tmp/LongCat-Audio-Codec')
os.chdir('/tmp/LongCat-Audio-Codec')  # Needed for relative paths in LongCat
import torch, soundfile as sf, subprocess, numpy as np
from scipy import signal
from networks.semantic_codec.model_loader import load_encoder, load_decoder

device = torch.device('cpu')
SR = 24000  # LongCat 24k decoder

BASE = '/tmp/LongCat-Audio-Codec'
# Load models
print("Loading LongCat...")
encoder = load_encoder(f'{BASE}/configs/LongCatAudioCodec_encoder.yaml', device)
# Model already loaded by load_encoder
decoder = load_decoder(f'{BASE}/configs/LongCatAudioCodec_decoder_24k_4codebooks.yaml', device)

# Load CMVN
cmvn = np.load(f'{BASE}/ckpts/LongCatAudioCodec_encoder_cmvn.npy')

def load_audio(path, dur=2):
    d, sr = sf.read(path)
    if sr != SR: d = signal.resample(d, int(len(d)*SR/sr), axis=0)
    L = dur*SR; d = d[:L]
    if d.ndim > 1: d = d.mean(axis=1)
    return torch.from_numpy(d).float().unsqueeze(0).unsqueeze(0)

base = '/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed'
src = load_audio(f'{base}/p255/p255_001_mic1.flac')
tgt = load_audio(f'{base}/p226/p226_001_mic1.flac')

# Encode (encoder handles CMVN internally)
with torch.no_grad():
    sem_src, aco_src = encoder(src)
    sem_tgt, aco_tgt = encoder(tgt)
    print(f"Semantic: {sem_src.shape}, Acoustic: {aco_src.shape}")

# Token swap: source semantic + target acoustic
sem_vc = sem_src
aco_vc = aco_tgt
# Trim to same length
T = min(sem_vc.shape[1], aco_vc.shape[1])
sem_vc = sem_vc[:, :T]
aco_vc = aco_vc[:, :T]

# Decode
audio_vc = decoder(sem_vc, aco_vc)
# Also decode original for comparison
audio_src = decoder(sem_src[:, :T], aco_src[:, :T])
audio_tgt = decoder(sem_tgt[:, :T], aco_tgt[:, :T])

out = '/Users/asill/research5'
sf.write(f'{out}/longcat_swap.wav', audio_vc.detach().squeeze().numpy(), SR)
sf.write(f'{out}/longcat_src.wav', audio_src.detach().squeeze().numpy(), SR)
sf.write(f'{out}/longcat_tgt.wav', audio_tgt.detach().squeeze().numpy(), SR)

# Cosine metrics
with torch.no_grad():
    sem_vc2, aco_vc2 = encoder(audio_vc)
    sem_src2, _ = encoder(audio_src)
    sem_tgt2, _ = encoder(audio_tgt)
    T2 = min(sem_vc2.shape[1], sem_src2.shape[1], sem_tgt2.shape[1])
    cs = torch.nn.functional.cosine_similarity(
        sem_vc2[:,:T2].float().reshape(-1), sem_src2[:,:T2].float().reshape(-1), dim=0)
    ct = torch.nn.functional.cosine_similarity(
        sem_vc2[:,:T2].float().reshape(-1), sem_tgt2[:,:T2].float().reshape(-1), dim=0)

print(f"Token Swap VC (p255→p226):")
print(f"  cos_src: {cs.item():.4f}")
print(f"  cos_tgt: {ct.item():.4f}")
print(f"  Δ: {ct.item()-cs.item():+.4f}")
print(f"✅ {out}/longcat_*.wav")
