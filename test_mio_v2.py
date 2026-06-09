#!/usr/bin/env python3
"""MioCodec: upper bound VC test + structure audit."""
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
cfg=model.config
print("Loaded in {:.1f}s".format(time.time()-t0))
print("SR={}Hz, hop={}, n_fft={}, downsample={}".format(cfg.sample_rate,cfg.hop_length,cfg.n_fft,cfg.downsample_factor))

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

print("Source: {:.1f}s, Target: {:.1f}s".format(len(d_src)/SR,len(d_tgt)/SR))

x_src=torch.from_numpy(d_src).float().unsqueeze(0)
x_tgt=torch.from_numpy(d_tgt).float().unsqueeze(0)

with torch.inference_mode():
    t0=time.time()
    feat_src=model.encode(x_src,return_content=True,return_global=True)
    te=time.time()-t0
    
    t0=time.time()
    feat_tgt=model.encode(x_tgt,return_content=True,return_global=True)
    te2=time.time()-t0
    
    ce=feat_src.content_embedding
    ct=feat_src.content_token_indices
    ge_src=feat_src.global_embedding
    ge_tgt=feat_tgt.global_embedding
    content_rate=ct.shape[0]/(len(d_src)/SR)
    
    print()
    print("=== STRUCTURE ===")
    print("Content embedding: {} (B=1, T={}, D={})".format(list(ce.shape),ce.shape[0],ce.shape[1]))
    print("Content tokens: {} (T={})".format(list(ct.shape),ct.shape[0]))
    print("Content cardinality: {}".format(ct.max().item()+1))
    print("Global embedding: {} (D={})".format(list(ge_src.shape),ge_src.shape[0]))
    print("Content rate: {:.0f} Hz (1 frame = {:.0f}ms)".format(content_rate,1000/content_rate))
    print("Encode time: {:.2f}s (source) + {:.2f}s (target)".format(te,te2))
    print("Global norm: src={:.2f}, tgt={:.2f}".format(ge_src.norm(),ge_tgt.norm()))
    print("cos(ge_src, ge_tgt): {:.4f}".format(
        torch.nn.functional.cosine_similarity(ge_src.unsqueeze(0),ge_tgt.unsqueeze(0)).item()))
    
    # Self-recon
    t0=time.time()
    mel_self=model.decode(global_embedding=ge_src,content_token_indices=ct,
                         target_audio_length=len(d_src))
    td=time.time()-t0
    print()
    print("Self-recon mel: {} time={:.2f}s".format(list(mel_self.shape),td))
    
    # VC
    t0=time.time()
    mel_vc=model.decode(global_embedding=ge_tgt,content_token_indices=ct,
                       target_audio_length=len(d_src))
    print("VC mel: {} time={:.2f}s".format(list(mel_vc.shape),time.time()-t0))
    
    # voice_conversion
    t0=time.time()
    mel_vcm=model.voice_conversion(x_src,x_tgt)
    print("voice_conversion mel: {} time={:.2f}s".format(list(mel_vcm.shape),time.time()-t0))
    
    # Mel difference
    mel_diff=(mel_vc-mel_self).abs().mean()
    print("Mel delta (VC - self): {:.4f}".format(mel_diff.item()))

# Save
np.save('/tmp/mio_self_mel.npy',mel_self.squeeze(0).cpu().numpy())
np.save('/tmp/mio_vc_mel.npy',mel_vc.squeeze(0).cpu().numpy())
np.save('/tmp/mio_src.npy',d_src)
np.save('/tmp/mio_tgt.npy',d_tgt)
print()
print("Saved to /tmp/mio_*.npy")

# Latency estimate
print()
print("=== LATENCY ESTIMATE ===")
print("Encode: {:.0f}ms".format(te*1000))
print("Decode: {:.0f}ms".format(td*1000))
print("Content frame: {:.0f}ms".format(1000/content_rate))
print("Total (offline): {:.0f}ms".format((te+td)*1000))
print("Note: WavLM + full-seq attention = non-streaming")
print("Conclusion: MioCodec is OFFLINE-ONLY for encode AND decode")
