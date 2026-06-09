#!/usr/bin/env python3
"""Pre-compute causal mel features for teacher dataset. Saves disk space + trains 100x faster."""
import torch, torchaudio, numpy as np, os
from scipy import signal as scipy_signal

SR=44100; TARGET_SR=16000; HOP_MS=20; N_MELS=80; N_FFT=512

DATA_DIR="/Users/asill/btrv5/data/mio_teacher"
MEL_DIR="/Users/asill/btrv5/data/mio_mel"
os.makedirs(MEL_DIR,exist_ok=True)

meta=np.load("{}/meta.npz".format(DATA_DIR))
n_samples=len(meta['spk_names'])
print("Pre-computing mel for {} samples...".format(n_samples))

mel_spec=torchaudio.transforms.MelSpectrogram(
    sample_rate=TARGET_SR,n_fft=N_FFT,hop_length=int(TARGET_SR*HOP_MS/1000),
    n_mels=N_MELS,f_min=80,f_max=7600,center=False,power=2)

for i in range(n_samples):
    d=np.load("{}/sample_{:04d}.npz".format(DATA_DIR,i))
    audio=d['audio']
    # Resample to 16k
    if len(audio)>0:
        audio_16k=scipy_signal.resample(audio,int(len(audio)*TARGET_SR/SR))
    else:
        audio_16k=audio
    x=torch.from_numpy(audio_16k).float().view(1,1,-1)
    with torch.no_grad():
        mel=mel_spec(x).squeeze(1)  # (1,80,T)
        logmel=torch.log(mel.clamp(min=1e-5)).squeeze(0)  # (80,T)
    
    np.savez_compressed("{}/mel_{:04d}.npz".format(MEL_DIR,i),
        logmel=logmel.numpy(),
        fsq_5d=d['fsq_5d'],
        fsq_tokens=d['fsq_tokens'],
        ce_768=d['ce_768'])
    
    if i%50==0: print("  {}/{}".format(i,n_samples))

print("Done! Mel features saved to {}/".format(MEL_DIR))
print("Sample shape: (80, T_50hz)")
