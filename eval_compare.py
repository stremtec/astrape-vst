#!/usr/bin/env python3
"""Fixed-metric evaluation: FiLM vs Transformer adapter VC."""
import sys, os
sys.path.insert(0, '/Users/asill/btrv5')
from mimi_splitter_v2 import load_mimi, MimiSplitterV2, mimi_encode, mimi_decode_latent, ContentExtractor, SpeakerEncoder
from transformer_adapter import TransformerAcousticAdapter, create_transformer_splitter
import torch, soundfile as sf, numpy as np
from scipy import signal
import whisper, glob
from scipy.signal import stft

device = torch.device('cpu')
SR = 24000
SAFE_LEN = 48000

mimi = load_mimi(device).to(device)

# Load both models
splitter_film = MimiSplitterV2(mimi).to(device)
splitter_film.load_state_dict(torch.load("checkpoints/mimi_splitter_v2_60spk.pt", map_location='cpu')['model_state_dict'])
splitter_film.eval()

splitter_trans = create_transformer_splitter(mimi, num_layers=2).to(device)
splitter_trans.load_state_dict(torch.load("checkpoints/mimi_transformer_adapter.pt", map_location='cpu')['model_state_dict'])
splitter_trans.eval()

model_w = whisper.load_model("base")
print("Models ready")

ROOT = "/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed"

def load_spk(spk):
    files = sorted(glob.glob(f"{ROOT}/{spk}/{spk}_*_mic1.flac"))
    d, sr = sf.read(files[0])
    if d.ndim > 1: d = d.mean(axis=1)
    if sr != SR: d = signal.resample(d, int(len(d)*SR/sr))
    if len(d) < SAFE_LEN: d = np.pad(d, (0, SAFE_LEN - len(d)))
    d = d[:SAFE_LEN].astype(np.float64)
    x = torch.from_numpy(d).float().view(1,1,-1).to(device)
    with torch.no_grad():
        z, codes = mimi_encode(x, mimi)
    return x, z, codes, d

def transcribe(audio_np):
    audio_f32 = audio_np.astype(np.float32)
    audio_f32 = audio_f32 / (np.abs(audio_f32).max() + 1e-8)
    result = model_w.transcribe(audio_f32, language="en", fp16=False)
    return result['text']

def wer(ref, hyp):
    ref_w = ref.lower().split()
    hyp_w = hyp.lower().split()
    d_mat = np.zeros((len(ref_w)+1, len(hyp_w)+1))
    for i in range(len(ref_w)+1): d_mat[i,0] = i
    for j in range(len(hyp_w)+1): d_mat[0,j] = j
    for i in range(1, len(ref_w)+1):
        for j in range(1, len(hyp_w)+1):
            cost = 0 if ref_w[i-1] == hyp_w[j-1] else 1
            d_mat[i,j] = min(d_mat[i-1,j]+1, d_mat[i,j-1]+1, d_mat[i-1,j-1]+cost)
    return d_mat[len(ref_w), len(hyp_w)] / max(len(ref_w), 1)

def compute_speaker_sim(a, b):
    """Simple cosine similarity on mean-pooled spectrogram."""
    f1,_,Z1 = stft(a, fs=SR, nperseg=512, noverlap=256)
    f2,_,Z2 = stft(b, fs=SR, nperseg=512, noverlap=256)
    Ts = min(Z1.shape[1], Z2.shape[1])
    v1 = np.mean(np.abs(Z1[:,:Ts]), axis=1)
    v2 = np.mean(np.abs(Z2[:,:Ts]), axis=1)
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return cos

def compute_f0(a):
    """Simple F0 estimate via autocorrelation."""
    from scipy.signal import correlate
    a = a - np.mean(a)
    corr = correlate(a, a, mode='full')
    corr = corr[len(corr)//2:]
    corr = corr / (corr[0] + 1e-8)
    peaks = np.where(corr[1:] < corr[:-1])[0]
    if len(peaks) == 0:
        return 0
    peak = peaks[0] + 1
    if peak > 0 and peak < len(corr) - 1:
        f0 = SR / peak
        return f0 if 50 < f0 < 400 else 0
    return 0

def compute_lsd(a, b):
    f1,_,Z1 = stft(a, fs=SR, nperseg=512, noverlap=256)
    f2,_,Z2 = stft(b, fs=SR, nperseg=512, noverlap=256)
    Ts = min(Z1.shape[1], Z2.shape[1])
    return np.mean(np.sqrt(np.mean(
        (np.log10(np.abs(Z1[:,:Ts])+1e-8)-np.log10(np.abs(Z2[:,:Ts])+1e-8))**2, axis=0)))*20

def vc(splitter, x_src, z_src, codes_src, z_tgt):
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
    return x_vc

pairs = [
    ("p255", "p226", "m->f"),
    ("p255", "p285", "m->f2"),
    ("p285", "p255", "f->m"),
]

print()
print("=" * 80)
print("  FiLM vs Transformer Adapter — Fixed Metrics")
print("=" * 80)

for src_spk, tgt_spk, desc in pairs:
    print()
    print("--- " + desc + ": " + src_spk + " -> " + tgt_spk + " ---")
    x_src, z_src, codes_src, d_src = load_spk(src_spk)
    x_tgt, z_tgt, codes_tgt, d_tgt = load_spk(tgt_spk)

    src_text = transcribe(d_src)
    tgt_text = transcribe(d_tgt)
    print("  SRC text: " + src_text)
    print("  TGT text: " + tgt_text)

    # Compute source→target sim for ΔSIM
    sim_src_tgt = compute_speaker_sim(d_src, d_tgt)
    sim_src_src = compute_speaker_sim(d_src, d_src)

    # FiLM VC
    x_vc_film = vc(splitter_film, x_src, z_src, codes_src, z_tgt)
    vc_film_np = x_vc_film[0,0].cpu().numpy()

    # Transformer VC
    x_vc_trans = vc(splitter_trans, x_src, z_src, codes_src, z_tgt)
    vc_trans_np = x_vc_trans[0,0].cpu().numpy()

    # Mimi RT
    x_rt = mimi_decode_latent(mimi, z_src)
    rt_np = x_rt[0,0].cpu().numpy()

    # Metrics
    wer_rt = wer(src_text, transcribe(rt_np))
    wer_film = wer(src_text, transcribe(vc_film_np))
    wer_trans = wer(src_text, transcribe(vc_trans_np))

    sim_film_tgt = compute_speaker_sim(vc_film_np, d_tgt)
    sim_trans_tgt = compute_speaker_sim(vc_trans_np, d_tgt)
    sim_film_src = compute_speaker_sim(vc_film_np, d_src)
    sim_trans_src = compute_speaker_sim(vc_trans_np, d_src)

    lsd_rt = compute_lsd(d_src, rt_np)
    lsd_film = compute_lsd(d_src, vc_film_np)
    lsd_trans = compute_lsd(d_src, vc_trans_np)

    f0_src = compute_f0(d_src)
    f0_tgt = compute_f0(d_tgt)
    f0_film = compute_f0(vc_film_np)
    f0_trans = compute_f0(vc_trans_np)

    # ΔSIM
    dsim_film = sim_film_tgt - sim_src_tgt
    dsim_trans = sim_trans_tgt - sim_src_tgt

    print()
    print("  " + "=" * 60)
    print("  Metric         MiMi-RT    FiLM      Transformer")
    print("  " + "-" * 60)
    print("  WER (%)        " + str(round(wer_rt*100,1)).rjust(6) +
          "    " + str(round(wer_film*100,1)).rjust(6) +
          "    " + str(round(wer_trans*100,1)).rjust(6))
    print("  SIM→tgt        " + "-".rjust(6) +
          "    " + str(round(sim_film_tgt,3)).rjust(6) +
          "    " + str(round(sim_trans_tgt,3)).rjust(6))
    print("  SIM→src        " + "-".rjust(6) +
          "    " + str(round(sim_film_src,3)).rjust(6) +
          "    " + str(round(sim_trans_src,3)).rjust(6))
    print("  ΔSIM→tgt       " + "-".rjust(6) +
          "    " + str(round(dsim_film,3)).rjust(6) +
          "    " + str(round(dsim_trans,3)).rjust(6))
    print("  LSD (dB)       " + str(round(lsd_rt,1)).rjust(6) +
          "    " + str(round(lsd_film,1)).rjust(6) +
          "    " + str(round(lsd_trans,1)).rjust(6))
    print("  F0 (Hz)        " + str(round(f0_src,0)).rjust(6) +
          " (" + str(round(f0_tgt,0)) + ")" +
          "    " + str(round(f0_film,0)).rjust(6) +
          "    " + str(round(f0_trans,0)).rjust(6))

    sf.write("checkpoints/vc_trans_" + src_spk + "_to_" + tgt_spk + ".wav", vc_trans_np, SR)

print()
print("Done!")
