#!/usr/bin/env python3
"""MioCodec upper bound + structure test."""
import torch, time, numpy as np, soundfile as sf
from scipy import signal
import sys
sys.path.insert(0, '/Users/asill/btrvrc0/.venv/lib/python3.12/site-packages')
from miocodec.model import MioCodecModel

SR=44100
print("Loading MioCodec...")
t0=time.time()
model=MioCodecModel.from_pretrained('Aratako/MioCodec-25Hz-44.1kHz-v2')
model.eval()
print("Loaded in {:.1f}s, SR={}Hz, content_rate={}Hz".format(time.time()-t0, model.sample_rate, model.content_rate))

src_path='/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed/p255/p255_001_mic1.flac'
tgt_path='/Users/asill/Downloads/origin.mp3'

d_src,sr_src=sf.read(src_path)
if d_src.ndim>1: d_src=d_src.mean(axis=1)
if sr_src!=SR: d_src=signal.resample(d_src,int(len(d_src)*SR/sr_src))
d_src=d_src[:SR*3]

d_tgt,sr_tgt=sf.read(tgt_path)
if d_tgt.ndim>1: d_tgt=d_tgt.mean(axis=1)
if sr_tgt!=SR: d_tgt=signal.resample(d_tgt,int(len(d_tgt)*SR/sr_tgt))
d_tgt=d_tgt[:SR*3]

print("Source: {:.1f}s, Target: {:.1f}s".format(len(d_src)/SR, len(d_tgt)/SR))

x_src=torch.from_numpy(d_src).float().unsqueeze(0)
x_tgt=torch.from_numpy(d_tgt).float().unsqueeze(0)

with torch.inference_mode():
    t0=time.time()
    feat_src=model.encode(x_src,return_content=True,return_global=True)
    print("Encode source: {:.2f}s".format(time.time()-t0))
    
    t0=time.time()
    feat_tgt=model.encode(x_tgt,return_content=True,return_global=True)
    print("Encode target: {:.2f}s".format(time.time()-t0))
    
    ce=feat_src.content_embedding
    ct=feat_src.content_token_indices
    ge_src=feat_src.global_embedding
    ge_tgt=feat_tgt.global_embedding
    
    print()
    print("Content embedding shape:", list(ce.shape))
    print("Content tokens shape:", list(ct.shape), "range:", ct.min().item(), "-", ct.max().item())
    print("Global embedding shape:", list(ge_src.shape))
    print("Global norm src: {:.2f}, tgt: {:.2f}".format(ge_src.norm(), ge_tgt.norm()))
    print("cos(ge_src, ge_tgt): {:.4f}".format(
        torch.nn.functional.cosine_similarity(ge_src, ge_tgt).item()))
    
    # Self-recon
    mel_self=model.decode(global_embedding=ge_src,content_token_indices=ct,
                         target_audio_length=len(d_src))
    print("Self-recon mel:", list(mel_self.shape))
    
    # VC
    mel_vc=model.decode(global_embedding=ge_tgt,content_token_indices=ct,
                       target_audio_length=len(d_src))
    print("VC mel:", list(mel_vc.shape))
    
    # voice_conversion method
    t0=time.time()
    mel_vcm=model.voice_conversion(x_src,x_tgt)
    print("voice_conversion mel:", list(mel_vcm.shape), "time: {:.2f}s".format(time.time()-t0))

# Save mels for analysis
np.save('/tmp/mio_self_mel.npy', mel_self.squeeze(0).cpu().numpy())
np.save('/tmp/mio_vc_mel.npy', mel_vc.squeeze(0).cpu().numpy())
np.save('/tmp/mio_src_audio.npy', d_src)
np.save('/tmp/mio_tgt_audio.npy', d_tgt)
print()
print("Saved mel spectrograms to /tmp/mio_*.npy")
print("Done!")
