#!/usr/bin/env python3
"""Mimi + FiLM latency on MPS with moshi-internal conv patch."""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import sys, time
sys.path.insert(0, '/Users/asill/btrv5')

# ── Patch moshi's internal conv before loading Mimi ────────────────────
import torch
import torch.nn.functional as F
_orig = F.conv1d

def _mps_safe_conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if input.device.type == 'mps' and weight.shape[0] > 60000:
        return _orig(input.to('cpu'), weight.to('cpu'), 
                     bias.to('cpu') if bias is not None else None,
                     stride, padding, dilation, groups).to('mps')
    return _orig(input, weight, bias, stride, padding, dilation, groups)

# Patch torch.nn.functional (used by moshi internally)
torch.nn.functional.conv1d = _mps_safe_conv1d
# Also patch the conv module's F reference
import moshi.modules.conv as mconv
mconv.F.conv1d = _mps_safe_conv1d

print("Patched moshi conv1d for MPS >65536 channel limit")

# ── Load ──────────────────────────────────────────────────────────────
from mimi_splitter_v2 import load_mimi, MimiSplitterV2, mimi_encode, mimi_decode_latent
import soundfile as sf, numpy as np
from scipy import signal

device = torch.device('mps')
print("Device:", device)

mimi = load_mimi(device).to(device)
splitter = MimiSplitterV2(mimi, n_content=1).to(device)
splitter.load_state_dict(torch.load("checkpoints/mimi_splitter_v2_60spk.pt", map_location=device)['model_state_dict'])
splitter.eval()

SR=24000; CHUNK_MS=80; CHUNK_SAMPLES=int(SR*CHUNK_MS/1000); STRIDE=1920
ROOT="/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed"
d,sr=sf.read(f"{ROOT}/p255/p255_001_mic1.flac")
if d.ndim>1: d=d.mean(axis=1)
if sr!=SR: d=signal.resample(d,int(len(d)*SR/sr))
n_chunks=len(d)//CHUNK_SAMPLES
chunks=[d[i*CHUNK_SAMPLES:(i+1)*CHUNK_SAMPLES] for i in range(n_chunks)]
print("Chunks:",n_chunks)

# Warmup
print("Warmup...")
with torch.no_grad():
    x=torch.from_numpy(chunks[0]).float().view(1,1,-1).to(device)
    _=mimi_encode(x,mimi)
torch.mps.synchronize()
# Target speaker
d_tgt,sr_tgt=sf.read("/Users/asill/Downloads/origin.mp3")
if d_tgt.ndim>1: d_tgt=d_tgt.mean(axis=1)
if sr_tgt!=SR: d_tgt=signal.resample(d_tgt,int(len(d_tgt)*SR/sr_tgt))
safe_tgt=(len(d_tgt)//STRIDE)*STRIDE; d_tgt=d_tgt[:safe_tgt]
x_tgt=torch.from_numpy(d_tgt).float().view(1,1,-1).to(device)
with torch.no_grad(): z_tgt,_=mimi_encode(x_tgt,mimi); S_tgt=splitter.speaker_encoder(z_tgt)
torch.mps.synchronize()
time.sleep(0.3)

# Measure
te,ts,td=[],[],[]
print("Measuring...")
for i in range(n_chunks):
    x=torch.from_numpy(chunks[i]).float().view(1,1,-1).to(device)
    torch.mps.synchronize(); t0=time.perf_counter()
    with torch.no_grad(): z,codes=mimi_encode(x,mimi)
    torch.mps.synchronize(); te.append(time.perf_counter()-t0)
    
    torch.mps.synchronize(); t0=time.perf_counter()
    with torch.no_grad():
        mimi.set_num_codebooks(1); z_q0=mimi.decode_latent(codes[:,:1,:]); mimi.set_num_codebooks(8)
        C=splitter.content_extractor(z_q0); n_ac=codes.shape[1]-1
        mimi.set_num_codebooks(n_ac); z_ac=mimi.decode_latent(codes[:,1:,:]); mimi.set_num_codebooks(8)
        A=splitter.acoustic_adapter(z_ac,S_tgt,C); z_vc=C+A
    torch.mps.synchronize(); ts.append(time.perf_counter()-t0)
    
    torch.mps.synchronize(); t0=time.perf_counter()
    with torch.no_grad(): _=mimi_decode_latent(mimi,z_vc)
    torch.mps.synchronize(); td.append(time.perf_counter()-t0)

te=np.array(te[1:])*1000; ts=np.array(ts[1:])*1000; td=np.array(td[1:])*1000; tt=te+ts+td

print()
print("="*65)
print("  MPS LATENCY (per 80ms chunk)")
print("="*65)
print("  {:20s} {:>8s} {:>8s} {:>8s} {:>8s}".format("","mean","p50","p95","max"))
print("  "+"-"*50)
for name,vals in [("Mimi encode",te),("Splitter+FiLM",ts),("Mimi decode",td),("TOTAL",tt)]:
    print("  {:20s} {:7.1f}ms {:7.1f}ms {:7.1f}ms {:7.1f}ms".format(
        name,np.mean(vals),np.median(vals),np.percentile(vals,95),np.max(vals)))

rtf=np.sum(tt)/1000/(n_chunks*CHUNK_MS/1000)
print()
print("  RTF: {:.3f}".format(rtf))
print("  Effective latency: ~{}ms (80ms frame + {:.0f}ms proc)".format(80+int(np.median(tt)),np.median(tt)))
print()
cpu_enc,cpu_dec,cpu_tot=18.7,9.7,28.6
print("  CPU p50: encode={:.0f}ms decode={:.0f}ms total={:.0f}ms".format(cpu_enc,cpu_dec,cpu_tot))
print("  MPS p50: encode={:.0f}ms decode={:.0f}ms total={:.0f}ms".format(np.median(te),np.median(td),np.median(tt)))
print("  Speedup: {:.1f}x".format(cpu_tot/np.median(tt)))
