#!/usr/bin/env python3
"""MPS latency — Mimi encode/decode on CPU, splitter on MPS."""
import os, sys, time
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
sys.path.insert(0, '/Users/asill/btrv5')

import torch
from mimi_splitter_v2 import load_mimi, MimiSplitterV2, mimi_encode, mimi_decode_latent
import soundfile as sf, numpy as np
from scipy import signal

cpu = torch.device('cpu')
mps = torch.device('mps')

# Mimi on CPU (MPS channel limit), splitter on MPS
mimi = load_mimi(cpu).to(cpu)
splitter = MimiSplitterV2(mimi, n_content=1)
# Move only splitter params to MPS
splitter.content_extractor = splitter.content_extractor.to(mps)
splitter.speaker_encoder = splitter.speaker_encoder.to(mps)
splitter.acoustic_adapter = splitter.acoustic_adapter.to(mps)
splitter.load_state_dict(torch.load("checkpoints/mimi_splitter_v2_60spk.pt", map_location='cpu')['model_state_dict'])
# Move loaded weights to MPS
for p in splitter.parameters():
    p.data = p.data.to(mps)
splitter.eval()
print("Mimi: CPU | Splitter: MPS")

SR=24000; CHUNK_MS=80; CHUNK_SAMPLES=int(SR*CHUNK_MS/1000); STRIDE=1920
ROOT="/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed"
d,sr=sf.read(f"{ROOT}/p255/p255_001_mic1.flac")
if d.ndim>1: d=d.mean(axis=1)
if sr!=SR: d=signal.resample(d,int(len(d)*SR/sr))
n_chunks=len(d)//CHUNK_SAMPLES
chunks=[d[i*CHUNK_SAMPLES:(i+1)*CHUNK_SAMPLES] for i in range(n_chunks)]
print("Chunks:",n_chunks)

# Target speaker (CPU)
d_tgt,sr_tgt=sf.read("/Users/asill/Downloads/origin.mp3")
if d_tgt.ndim>1: d_tgt=d_tgt.mean(axis=1)
if sr_tgt!=SR: d_tgt=signal.resample(d_tgt,int(len(d_tgt)*SR/sr_tgt))
safe_tgt=(len(d_tgt)//STRIDE)*STRIDE; d_tgt=d_tgt[:safe_tgt]
x_tgt=torch.from_numpy(d_tgt).float().view(1,1,-1).to(cpu)
with torch.no_grad(): z_tgt,_=mimi_encode(x_tgt,mimi)

# Warmup
print("Warmup...")
with torch.no_grad():
    x=torch.from_numpy(chunks[0]).float().view(1,1,-1).to(cpu)
    _=mimi_encode(x,mimi)
torch.mps.synchronize()
time.sleep(0.3)

# Measure
te,ts,td=[],[],[]
print("Measuring...")
for i in range(n_chunks):
    x=torch.from_numpy(chunks[i]).float().view(1,1,-1).to(cpu)
    
    # Encode (CPU)
    t0=time.perf_counter()
    with torch.no_grad(): z,codes=mimi_encode(x,mimi)
    te.append(time.perf_counter()-t0)
    
    # Splitter (MPS): need to move data to MPS, process, move back
    torch.mps.synchronize(); t0=time.perf_counter()
    with torch.no_grad():
        z_mps=z.to(mps); codes_mps=codes.to(mps)
        mimi.set_num_codebooks(1); z_q0=mimi.decode_latent(codes_mps[:,:1,:]); mimi.set_num_codebooks(8)
        C=splitter.content_extractor(z_q0)
        S_tgt_mps=splitter.speaker_encoder(z_mps)
        n_ac=codes_mps.shape[1]-1
        mimi.set_num_codebooks(n_ac); z_ac=mimi.decode_latent(codes_mps[:,1:,:]); mimi.set_num_codebooks(8)
        A=splitter.acoustic_adapter(z_ac,S_tgt_mps,C); z_vc=C+A
    torch.mps.synchronize(); ts.append(time.perf_counter()-t0)
    
    # Decode (CPU): move z_vc back to CPU
    t0=time.perf_counter()
    with torch.no_grad():
        z_vc_cpu=z_vc.to(cpu)
        _=mimi_decode_latent(mimi,z_vc_cpu)
    td.append(time.perf_counter()-t0)

te=np.array(te[1:])*1000; ts=np.array(ts[1:])*1000
td=np.array(td[1:])*1000; tt=te+ts+td

print()
print("="*65)
print("  HYBRID LATENCY (Mimi=CPU, Splitter=MPS)")
print("="*65)
print("  {:20s} {:>8s} {:>8s} {:>8s} {:>8s}".format("","mean","p50","p95","max"))
print("  "+"-"*50)
for name,vals in [("Mimi encode (CPU)",te),("Splitter (MPS)",ts),("Mimi decode (CPU)",td),("TOTAL",tt)]:
    print("  {:20s} {:7.1f}ms {:7.1f}ms {:7.1f}ms {:7.1f}ms".format(
        name,np.mean(vals),np.median(vals),np.percentile(vals,95),np.max(vals)))

rtf=np.sum(tt)/1000/(n_chunks*CHUNK_MS/1000)
print()
print("  RTF: {:.3f}".format(rtf))
print("  Effective latency: ~{}ms".format(80+int(np.median(tt))))

print()
cpu_enc,cpu_dec,cpu_tot=18.7,9.7,28.6
print("  Pure CPU: encode={:.0f}ms decode={:.0f}ms total={:.0f}ms".format(cpu_enc,cpu_dec,cpu_tot))
print("  Hybrid:   encode={:.0f}ms splitter={:.0f}ms decode={:.0f}ms total={:.0f}ms".format(
    np.median(te),np.median(ts),np.median(td),np.median(tt)))
