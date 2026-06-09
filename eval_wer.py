#!/usr/bin/env python3
"""WER + speaker SIM evaluation for 60-spk Mimi Splitter."""
import sys, os
sys.path.insert(0, '/Users/asill/btrv5')
from mimi_splitter_v2 import load_mimi, MimiSplitterV2, mimi_encode, mimi_decode_latent
import torch, soundfile as sf, numpy as np
from scipy import signal
import whisper, glob

device = torch.device('cpu')
SR = 24000
SAFE_LEN = 48000

mimi = load_mimi(device).to(device)
splitter = MimiSplitterV2(mimi).to(device)
splitter.load_state_dict(torch.load("checkpoints/mimi_splitter_v2_60spk.pt", map_location='cpu')['model_state_dict'])
splitter.eval()
model_w = whisper.load_model("base")
print("Models ready")

def transcribe(audio_np):
    audio_f32 = audio_np.astype(np.float32)
    audio_f32 = audio_f32 / (np.abs(audio_f32).max() + 1e-8)
    result = model_w.transcribe(audio_f32, language="en", fp16=False)
    return result['text']

ROOT = "/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed"

def load_spk(spk):
    files = sorted(glob.glob(f"{ROOT}/{spk}/{spk}_*_mic1.flac"))
    d, sr = sf.read(files[0])
    if d.ndim > 1:
        d = d.mean(axis=1)
    if sr != SR:
        d = signal.resample(d, int(len(d)*SR/sr))
    if len(d) < SAFE_LEN:
        d = np.pad(d, (0, SAFE_LEN - len(d)))
    d = d[:SAFE_LEN].astype(np.float64)
    x = torch.from_numpy(d).float().view(1,1,-1).to(device)
    with torch.no_grad():
        z, codes = mimi_encode(x, mimi)
    return x, z, codes, d

def wer(ref, hyp):
    ref_w = ref.lower().split()
    hyp_w = hyp.lower().split()
    d_mat = np.zeros((len(ref_w)+1, len(hyp_w)+1))
    for i in range(len(ref_w)+1):
        d_mat[i,0] = i
    for j in range(len(hyp_w)+1):
        d_mat[0,j] = j
    for i in range(1, len(ref_w)+1):
        for j in range(1, len(hyp_w)+1):
            cost = 0 if ref_w[i-1] == hyp_w[j-1] else 1
            d_mat[i,j] = min(d_mat[i-1,j]+1, d_mat[i,j-1]+1, d_mat[i-1,j-1]+cost)
    return d_mat[len(ref_w), len(hyp_w)] / max(len(ref_w), 1)

pairs = [
    ("p255", "p285", "male->female"),
    ("p255", "p226", "male->female2"),
    ("p285", "p255", "female->male"),
]

print()
print("=" * 70)

for src_spk, tgt_spk, desc in pairs:
    print()
    print("--- " + desc + ": " + src_spk + " -> " + tgt_spk + " ---")
    x_src, z_src, codes_src, d_src = load_spk(src_spk)
    x_tgt, z_tgt, codes_tgt, d_tgt = load_spk(tgt_spk)

    src_text = transcribe(d_src)
    print("  SRC text: " + src_text)

    # VC
    with torch.no_grad():
        mimi.set_num_codebooks(1)
        z_q0 = mimi.decode_latent(codes_src[:, :1, :])
        mimi.set_num_codebooks(8)
        C = splitter.content_extractor(z_q0)
        S = splitter.speaker_encoder(z_tgt)
        n_ac = codes_src.shape[1] - 1
        mimi.set_num_codebooks(n_ac)
        z_ac = mimi.decode_latent(codes_src[:, 1:, :])
        mimi.set_num_codebooks(8)
        A = splitter.acoustic_adapter(z_ac, S, C)
        z_vc = C + A
        x_vc = mimi_decode_latent(mimi, z_vc)

    vc_text = transcribe(x_vc[0,0].cpu().numpy())
    print("  VC  text: " + vc_text)

    x_rt = mimi_decode_latent(mimi, z_src)
    rt_text = transcribe(x_rt[0,0].cpu().numpy())

    w_wer = wer(src_text, vc_text) * 100
    rt_wer_score = wer(src_text, rt_text) * 100
    print("  VC  WER: " + str(round(w_wer, 1)) + "%")
    print("  RT  WER: " + str(round(rt_wer_score, 1)) + "%")

    # Speaker cosine
    with torch.no_grad():
        _, _, S_vc, _ = splitter(z_vc, codes_src)
        cos_sim = torch.nn.functional.cosine_similarity(S, S_vc)
    print("  Speaker cos: " + str(round(cos_sim.item(), 3)))

    sf.write("checkpoints/vc_" + src_spk + "_to_" + tgt_spk + ".wav", x_vc[0,0].cpu().numpy(), SR)

print()
print("Done!")
