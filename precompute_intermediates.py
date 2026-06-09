#!/usr/bin/env python3
"""Precompute teacher intermediate features for v2 distillation."""
import torch, numpy as np, os, time
from miocodec.model import MioCodecModel

SR=44100
teacher=MioCodecModel.from_pretrained('Aratako/MioCodec-25Hz-44.1kHz-v2'); teacher.eval()

DATA_DIR="/Users/asill/btrv5/data/mio_teacher"
INTER_DIR="/Users/asill/btrv5/data/mio_intermediate"
os.makedirs(INTER_DIR,exist_ok=True)

meta=np.load("{}/meta.npz".format(DATA_DIR)); n=len(meta['spk_names'])
print("Extracting teacher intermediates for {} samples...".format(n))

for idx in range(n):
    d=np.load("{}/sample_{:04d}.npz".format(DATA_DIR,idx))
    audio=d['audio']; alen=len(audio)
    x=torch.from_numpy(audio[:SR*3]).float().unsqueeze(0)
    
    with torch.inference_mode():
        # Full encode to get content embedding
        feat=teacher.encode(x,return_content=True,return_global=True)
        ce_768=feat.content_embedding  # (T,768)
        ct=feat.content_token_indices  # (T,)
        ge_128=feat.global_embedding   # (128,)
        
        # Internal: get pre-FSQ latent (before quantization)
        padding=teacher._calculate_waveform_padding(len(audio[:SR*3]))
        local_ssl,global_ssl=teacher.forward_ssl_features(x,padding=padding)
        
        # Local SSL features at 50Hz (WavLM layers 6/9 averaged)
        local_ssl_768=local_ssl.squeeze(0)  # (T50,768)
        
        # Local encoder → pre-FSQ (before quantizer's proj_in+quantize+proj_out)
        local_enc=teacher.local_encoder(local_ssl)  # (1,T50,768)
        local_enc=teacher.conv_downsample(local_enc.transpose(1,2)).transpose(1,2)  # (1,T25,768)
        pre_fsq_768=local_enc.squeeze(0)  # (T25,768)
        
        # FSQ 5d (pre-quantization scalar)
        fsq_5d=teacher.local_quantizer.proj_in(local_enc).squeeze(0)  # (T25,5)
    
    np.savez_compressed("{}/inter_{:04d}.npz".format(INTER_DIR,idx),
        ce_768=ce_768.numpy(), ct=ct.numpy(), ge_128=ge_128.numpy(),
        local_ssl_768=local_ssl_768.numpy(), pre_fsq_768=pre_fsq_768.numpy(),
        fsq_5d=fsq_5d.numpy())
    
    if idx%50==0: print("  {}/{}".format(idx,n))

print("Done! Saved to {}/".format(INTER_DIR))
print("Fields: ce_768, ct, ge_128, local_ssl_768, pre_fsq_768, fsq_5d")
print("Shapes: local_ssl=(T50,768), pre_fsq=(T25,768), ce_768=(T25,768), fsq_5d=(T25,5)")
