#!/usr/bin/env python3
"""MPS latency — override specific SEANet conv to CPU."""
import os, sys, time
sys.path.insert(0, '/Users/asill/btrv5')

import torch
import torch.nn as nn

# ── Strategy: find the specific conv that fails and wrap it ──────────────
# The failing conv is in Moshi's SEANet encoder — we'll patch after loading

from mimi_splitter_v2 import load_mimi, MimiSplitterV2, mimi_encode, mimi_decode_latent
import soundfile as sf, numpy as np
from scipy import signal

device = torch.device('mps')
cpu = torch.device('cpu')
print("Device:", device)

# Load Mimi on CPU first to inspect
mimi_cpu = load_mimi(cpu)
mimi_cpu.eval()

# Find ALL conv1d layers with out_channels > 60000
big_convs = []
for name, mod in mimi_cpu.named_modules():
    if isinstance(mod, nn.Conv1d) and mod.out_channels > 60000:
        big_convs.append(name)
        print(f"  Found: {name} out_ch={mod.out_channels}")

if not big_convs:
    print("  No large conv found — checking other patterns...")
    for name, mod in mimi_cpu.named_modules():
        if isinstance(mod, nn.Conv1d):
            if mod.out_channels > 10000:
                print(f"    {name}: out_ch={mod.out_channels}, in_ch={mod.in_channels}, groups={mod.groups}")

# Actually, let me just wrap ALL conv layers that have issues on MPS
# by creating a custom wrapper

class MPSSafeConv1d(nn.Conv1d):
    """Conv1d that falls back to CPU when MPS can't handle it."""
    def _conv_forward(self, input, weight, bias):
        if input.device.type == 'mps' and self.out_channels > 60000:
            inp = input.to('cpu')
            w = weight.to('cpu')
            b = bias.to('cpu') if bias is not None else None
            result = super()._conv_forward(inp, w, b)
            return result.to('mps')
        return super()._conv_forward(input, weight, bias)

# Wrap Mimi model's conv layers
def wrap_mimi_conv(module, path=''):
    for name, child in list(module.named_children()):
        full = f"{path}.{name}" if path else name
        if isinstance(child, nn.Conv1d) and child.out_channels > 60000:
            # Replace with safe wrapper
            safe = MPSSafeConv1d(
                child.in_channels, child.out_channels, child.kernel_size,
                child.stride, child.padding, child.dilation,
                child.groups, child.bias is not None, child.padding_mode
            )
            safe.weight.data = child.weight.data
            if child.bias is not None:
                safe.bias.data = child.bias.data
            setattr(module, name, safe)
            print(f"  Wrapped: {full} (out_ch={child.out_channels})")
        else:
            wrap_mimi_conv(child, full)

# Load Mimi fresh and wrap
mimi = load_mimi(device)
mimi.eval()
wrap_mimi_conv(mimi)
print("Mimi conv layers wrapped")

# Now load splitter
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

# Target
d_tgt,sr_tgt=sf.read("/Users/asill/Downloads/origin.mp3")
if d_tgt.ndim>1: d_tgt=d_tgt.mean(axis=1)
if sr_tgt!=SR: d_tgt=signal.resample(d_tgt,int(len(d_tgt)*SR/sr_tgt))
safe_tgt=(len(d_tgt)//STRIDE)*STRIDE; d_tgt=d_tgt[:safe_tgt]
x_tgt=torch.from_numpy(d_tgt).float().view(1,1,-1).to(device)
print("Target encode...")
with torch.no_grad(): z_tgt,_=mimi_encode(x_tgt,mimi); S_tgt=splitter.speaker_encoder(z_tgt)
torch.mps.synchronize()
print("OK")

# Warmup
with torch.no_grad():
    x=torch.from_numpy(chunks[0]).float().view(1,1,-1).to(device)
    _=mimi_encode(x,mimi)
torch.mps.synchronize(); time.sleep(0.3)

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
print("  Latency: ~{}ms".format(80+int(np.median(tt))))

print()
cpu_enc,cpu_dec,cpu_tot=18.7,9.7,28.6
mps_enc=np.median(te); mps_dec=np.median(td); mps_tot=np.median(tt)
print("  CPU: encode={:.0f}ms decode={:.0f}ms total={:.0f}ms".format(cpu_enc,cpu_dec,cpu_tot))
print("  MPS: encode={:.0f}ms decode={:.0f}ms total={:.0f}ms".format(mps_enc,mps_dec,mps_tot))
print("  Speedup: {:.1f}x".format(cpu_tot/mps_tot))
