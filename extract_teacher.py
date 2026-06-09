#!/usr/bin/env python3
"""MioCodec Teacher Extraction: FSQ scalars + tokens + embeddings + global."""
import torch, numpy as np, soundfile as sf, os, glob, time
from scipy import signal
from miocodec.model import MioCodecModel

SR=44100
model=MioCodecModel.from_pretrained('Aratako/MioCodec-25Hz-44.1kHz-v2')
model.eval()

SPKS=['p225','p226','p227','p228','p229','p230','p231','p232','p233','p234',
      'p236','p237','p238','p239','p240','p241','p243','p244','p245','p246',
      'p247','p248','p249','p250','p251','p252','p253','p254','p255','p256',
      'p257','p258','p259','p260','p261','p262','p263','p264','p265','p266'][:30]
ROOT="/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed"

OUT_DIR="/Users/asill/btrv5/data/mio_teacher"
os.makedirs(OUT_DIR,exist_ok=True)

print("Extracting teacher data for {} speakers...".format(len(SPKS)))
total_samples=0

all_data=[]

for spk_idx,spk in enumerate(SPKS):
    files=sorted(glob.glob("{}/{}/{}_*_mic1.flac".format(ROOT,spk,spk)))
    for fi,f in enumerate(files[:5]):
        d,s=sf.read(f)
        if d.ndim>1: d=d.mean(axis=1)
        if s!=SR: d=signal.resample(d,int(len(d)*SR/s))
        d=d[:SR*3]  # 3 seconds
        d=d[:(len(d)//model.config.hop_length)*model.config.hop_length]
        
        x=torch.from_numpy(d).float().unsqueeze(0)
        
        with torch.inference_mode():
            # Full encode
            feat=model.encode(x,return_content=True,return_global=True)
            
            # Get FSQ intermediate: manual forward
            padding=model._calculate_waveform_padding(len(d))
            local_ssl,global_ssl=model.forward_ssl_features(x,padding=padding)
            local_enc=model.local_encoder(local_ssl)
            local_enc=model.conv_downsample(local_enc.transpose(1,2)).transpose(1,2)
            
            # FSQ scalars (5-dim, pre-quantization)
            fsq_5d=model.local_quantizer.proj_in(local_enc).squeeze(0)  # (T,5)
            
            # FSQ quantized (bounded to [-1,1] per dim)
            z_q,indices=model.local_quantizer.fsq.encode(fsq_5d.unsqueeze(0))
            z_q=z_q.squeeze(0)  # (T,5)
            indices=indices.squeeze(0)  # (T,)
            
            # Content embedding (768d projection)
            ce_768=model.local_quantizer.proj_out(z_q.unsqueeze(0)).squeeze(0)  # (T,768)
            
            # Global embedding
            ge_128=feat.global_embedding  # (128,)
            
            # Teacher waveform (self-recon for quality reference)
            wav_teacher=model.decode(global_embedding=ge_128,
                                    content_token_indices=feat.content_token_indices,
                                    target_audio_length=len(d))
        
        sample={
            'spk': spk, 'spk_idx': spk_idx, 'file': os.path.basename(f),
            'fsq_5d': fsq_5d.cpu().numpy(),        # (T,5) target for regression
            'fsq_tokens': indices.cpu().numpy(),     # (T,) target for CE
            'ce_768': ce_768.cpu().numpy(),          # (T,768) reference
            'ge_128': ge_128.cpu().numpy(),          # (128,) global embedding
            'audio': d,                              # raw audio
            'audio_len': len(d),
            'content_len': fsq_5d.shape[0],          # T_content at 25Hz
        }
        all_data.append(sample)
        total_samples+=1
    
    print("  {}/{} {}: {} samples".format(spk_idx+1,len(SPKS),spk,5))

# Save compact dataset
print()
print("Saving dataset: {} samples".format(total_samples))

# Save as compressed npz (only essential tensors)
save_data={
    'meta': [(s['spk'],s['spk_idx'],s['file'],s['audio_len'],s['content_len']) for s in all_data],
}

# Stack tensors with padding (variable T_content)
max_T=max(s['content_len'] for s in all_data)
print("Max content length: {} frames ({}s at 25Hz)".format(max_T,max_T/25))

# Save individual files for flexibility
np.savez_compressed("{}/meta.npz".format(OUT_DIR),
    spk_names=np.array([s['spk'] for s in all_data]),
    spk_idxs=np.array([s['spk_idx'] for s in all_data]),
    audio_lens=np.array([s['audio_len'] for s in all_data]),
    content_lens=np.array([s['content_len'] for s in all_data]),
)

# Save tensors in shards
for i,s in enumerate(all_data):
    np.savez_compressed("{}/sample_{:04d}.npz".format(OUT_DIR,i),
        fsq_5d=s['fsq_5d'],
        fsq_tokens=s['fsq_tokens'],
        ce_768=s['ce_768'],
        ge_128=s['ge_128'],
        audio=s['audio'],
    )

# Global embedding cache per speaker
spk_ge={}
for s in all_data:
    spk=s['spk']
    if spk not in spk_ge:
        spk_ge[spk]=[]
    spk_ge[spk].append(s['ge_128'])

# Average per speaker
spk_ge_avg={spk: np.mean(np.stack(ge_list),axis=0) for spk,ge_list in spk_ge.items()}
np.savez_compressed("{}/spk_global_emb.npz".format(OUT_DIR),**spk_ge_avg)

print("Saved to {}/".format(OUT_DIR))
print("Files: meta.npz, sample_XXXX.npz ({} files), spk_global_emb.npz".format(total_samples))
print()
print("Dataset summary:")
print("  FSQ 5-dim scalars: target for regression student")
print("  FSQ token indices: target for CE student")
print("  Content embedding 768d: reference for quality check")
print("  Global embedding 128d: cache per speaker")
print("  Audio: raw waveform for reference")
print()
print("Done!")
