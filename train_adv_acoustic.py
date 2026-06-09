#!/usr/bin/env python3
"""
Adversarial Acoustic Adapter: strip source speaker from q1-q7, inject target.
Keeps n_content=1 (clean), improves acoustic path with adversarial training.
"""
import sys, os, glob, time, warnings
sys.path.insert(0, '/Users/asill/btrv5')
from mimi_splitter_v2 import load_mimi, MimiSplitterV2, mimi_encode, mimi_decode_latent
import torch, torch.nn as nn
import soundfile as sf, numpy as np
from scipy import signal
from torch.optim import AdamW
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
SR = 24000; SAFE_LEN = 48000; STRIDE = 1920
BATCH = 8; EPOCHS = 200

device = torch.device('cpu')
print("Device:", device, "| Adv acoustic adapter", flush=True)

mimi = load_mimi(device).to(device)

# ── Adversarial Acoustic Adapter ───────────────────────────────────────
class AdvAcousticAdapter(nn.Module):
    """Speaker-removal + target-injection for acoustic path."""
    def __init__(self, dim=512, spk_dim=256, hidden=384):
        super().__init__()
        self.spk_proj = nn.Linear(dim, spk_dim)  # project 512→256
        # Speaker removal encoder
        self.remove_conv = nn.Sequential(
            nn.Conv1d(dim, hidden, 5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, dim, 5, padding=2),
        )
        # Target speaker FiLM
        self.scale = nn.Sequential(nn.Linear(spk_dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.bias = nn.Sequential(nn.Linear(spk_dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        # Content-aware refinement
        self.refine = nn.Sequential(
            nn.Conv1d(dim*2, hidden, 3, padding=1), nn.GELU(),
            nn.Conv1d(hidden, dim, 3, padding=1),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, z_ac, spk_emb, z_content):
        spk = self.spk_proj(spk_emb)  # (B, spk_dim)
        # Remove source speaker
        h = self.remove_conv(z_ac)
        # Inject target speaker
        s = self.scale(spk).unsqueeze(-1)
        b = self.bias(spk).unsqueeze(-1)
        h = h * (1 + torch.tanh(s)) + b
        # Content-aware
        combined = torch.cat([h, z_content], dim=1)
        refined = self.refine(combined)
        # Norm
        out = refined.transpose(1,2)
        out = self.norm(out).transpose(1,2)
        return out


class MimiSplitterAdv(nn.Module):
    def __init__(self, mimi):
        super().__init__()
        self.mimi = mimi
        from mimi_splitter_v2 import ContentExtractor, SpeakerEncoder
        self.content_extractor = ContentExtractor(512, 64)
        self.speaker_encoder = SpeakerEncoder(512, 256)
        self.acoustic_adapter = AdvAcousticAdapter(512, 256)

    def _get_q0_qac(self, codes):
        with torch.no_grad():
            self.mimi.set_num_codebooks(1)
            z_q0 = self.mimi.decode_latent(codes[:, :1, :])
            n_ac = codes.shape[1] - 1
            self.mimi.set_num_codebooks(n_ac)
            z_ac = self.mimi.decode_latent(codes[:, 1:, :])
            self.mimi.set_num_codebooks(8)
        return z_q0, z_ac

    def forward(self, z_post, codes):
        z_q0, z_ac = self._get_q0_qac(codes)
        C = self.content_extractor(z_q0)
        S = self.speaker_encoder(z_post)
        A = self.acoustic_adapter(z_ac, S, C)
        z_vc = C + A
        return z_vc, C, S, A


# ── Data ────────────────────────────────────────────────────────────────
TRAIN_SPKS = ['p225','p226','p227','p228','p229','p230','p231','p232','p233','p234',
    'p236','p237','p238','p239','p240','p241','p243','p244','p245','p246',
    'p247','p248','p249','p250','p251','p252','p253','p254','p255','p256',
    'p257','p258','p259','p260','p261','p262','p263','p264','p265','p266'][:40]

splitter = MimiSplitterAdv(mimi).to(device)
print("Params:", sum(p.numel() for p in splitter.parameters() if p.requires_grad))

ROOT = "/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed"

def encode_speakers(spk_list, n_utt=5):
    samples = []
    for spk_idx, spk in enumerate(spk_list):
        files = sorted(glob.glob(f"{ROOT}/{spk}/{spk}_*_mic1.flac"))[:n_utt]
        for f in files:
            d, sr = sf.read(f)
            if d.ndim > 1: d = d.mean(axis=1)
            if sr != SR: d = signal.resample(d, int(len(d)*SR/sr))
            if len(d) < SAFE_LEN: d = np.pad(d, (0, SAFE_LEN - len(d)))
            d = d[:SAFE_LEN]
            x = torch.from_numpy(d).float().view(1,1,-1).to(device)
            with torch.no_grad():
                z, codes = mimi_encode(x, mimi)
            samples.append({'z':z.squeeze(0).cpu(),'codes':codes.squeeze(0).cpu(),'spk':spk_idx})
    return samples

print("Pre-encoding...", flush=True); t0=time.time()
train_data = encode_speakers(TRAIN_SPKS, 5)
print(f"Train:{len(train_data)} ({time.time()-t0:.1f}s)")

# ── Classifiers for adversarial ─────────────────────────────────────────
class SpkClf(nn.Module):
    def __init__(self, n_spk=40):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(512,256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256,n_spk))
    def forward(self, x):
        if x.dim()==3: x=x.mean(dim=-1)
        return self.net(x)

from torch.autograd import Function
class GR(Function):
    @staticmethod
    def forward(ctx,x,a): ctx.alpha=a; return x.view_as(x)
    @staticmethod
    def backward(ctx,g): return g.neg()*ctx.alpha,None

# ── Training with 3-way adversarial ─────────────────────────────────────
mse = nn.MSELoss(); ce = nn.CrossEntropyLoss()
opt = AdamW(splitter.parameters(), lr=1e-3, weight_decay=1e-5)
clf_C = SpkClf(len(TRAIN_SPKS)).to(device)
clf_S = SpkClf(len(TRAIN_SPKS)).to(device)
clf_A = SpkClf(len(TRAIN_SPKS)).to(device)  # adversarial on acoustic path
opt_clf = AdamW(list(clf_C.parameters())+list(clf_S.parameters())+list(clf_A.parameters()), lr=1e-3)

print()
print("Training (3-way adv: C→chance, S→100%, A→chance)...", flush=True)

for epoch in range(EPOCHS):
    idxs = torch.randperm(len(train_data))
    tr, tadv_c, tadv_a, tspk, nb = 0,0,0,0,0

    for i in range(0, len(train_data), BATCH):
        batch = [train_data[j] for j in idxs[i:i+BATCH]]
        zb = torch.stack([s['z'] for s in batch]).to(device)
        cb = torch.stack([s['codes'] for s in batch]).to(device)
        spk = torch.tensor([s['spk'] for s in batch]).to(device)

        zv, C, S, A = splitter(zb, cb)

        Lr = mse(zv, zb)
        Lc = ce(clf_C(GR.apply(C.mean(dim=-1), 1.0)), spk)        # C→chance
        La = ce(clf_A(GR.apply(A.mean(dim=-1), 0.3)), spk)        # A→chance (weaker adv)
        Ls = ce(clf_S(S), spk)                                     # S→100%

        loss = Lr + 0.5*Lc + 0.3*La + 0.5*Ls

        opt.zero_grad(); opt_clf.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(splitter.parameters(), 1.0)
        opt.step(); opt_clf.step()

        tr += Lr.item(); tadv_c += Lc.item(); tadv_a += La.item(); tspk += Ls.item(); nb += 1

    if epoch % 20 == 0 or epoch == EPOCHS - 1:
        n = max(nb,1)
        print(f"  E{epoch:3d} R={tr/n:.3f} C_adv={tadv_c/n:.2f} A_adv={tadv_a/n:.2f} S_spk={tspk/n:.2f}", flush=True)

os.makedirs("checkpoints", exist_ok=True)
torch.save({"model_state_dict": splitter.state_dict()}, "checkpoints/mimi_adv_ac.pt")

# ── VC test ────────────────────────────────────────────────────────────
print()
print("=== VC: p255 -> origin (Adv Acoustic) ===")
splitter.eval()

d_src, sr = sf.read(f"{ROOT}/p255/p255_001_mic1.flac")
if d_src.ndim>1: d_src=d_src.mean(axis=1)
if sr!=SR: d_src=signal.resample(d_src,int(len(d_src)*SR/sr))
safe_src=(len(d_src)//STRIDE)*STRIDE; d_src=d_src[:safe_src]; src_len=len(d_src)
x_src=torch.from_numpy(d_src).float().view(1,1,-1).to(device)
with torch.no_grad(): z_src,codes_src=mimi_encode(x_src,mimi)

d_tgt,sr=sf.read("/Users/asill/Downloads/origin.mp3")
if d_tgt.ndim>1: d_tgt=d_tgt.mean(axis=1)
if sr!=SR: d_tgt=signal.resample(d_tgt,int(len(d_tgt)*SR/sr))
safe_tgt=(len(d_tgt)//STRIDE)*STRIDE; d_tgt=d_tgt[:safe_tgt]
x_tgt=torch.from_numpy(d_tgt).float().view(1,1,-1).to(device)
with torch.no_grad(): z_tgt,codes_tgt=mimi_encode(x_tgt,mimi)

with torch.no_grad():
    z_q0,_ = splitter._get_q0_qac(codes_src)
    C = splitter.content_extractor(z_q0)
    S_tgt = splitter.speaker_encoder(z_tgt)
    _, z_ac = splitter._get_q0_qac(codes_src)
    A = splitter.acoustic_adapter(z_ac, S_tgt, C)
    z_vc = C + A
    x_vc = mimi_decode_latent(mimi, z_vc)

x_rt = mimi_decode_latent(mimi, z_src)
vc_np = x_vc[0,0].cpu().numpy()[:src_len]
rt_np = x_rt[0,0].cpu().numpy()[:src_len]

from scipy.signal import stft
def ana(a,label):
    f,_,Z=stft(a,fs=SR,nperseg=512,noverlap=256); mag=np.abs(Z); total=mag.sum()+1e-8
    c=np.sum(f[:len(f)//2,np.newaxis]*mag[:len(f)//2],axis=0)/(mag[:len(f)//2].sum(axis=0)+1e-8)
    vhigh=mag[(f>=4000)&(f<8000)].sum()/total*100
    low=mag[(f>=0)&(f<300)].sum()/total*100
    print(f"  {label:15s} RMS={np.sqrt(np.mean(a**2)):.3f} Cent={np.mean(c):.0f}Hz VHigh={vhigh:.1f}% Low={low:.1f}%")

print()
ana(d_src,"Source (p255)")
ana(d_tgt,"Target (origin)")
ana(vc_np,"VC adv acoustic")
ana(rt_np,"Mimi RT")

sf.write("/Users/asill/Desktop/vc_advac_p255_origin.wav", vc_np, SR)
sf.write("/Users/asill/Desktop/vc_advac_src.wav", d_src, SR)
sf.write("/Users/asill/Desktop/vc_advac_tgt.wav", d_tgt, SR)
print()
print("Saved: Desktop/vc_advac_*.wav")
print("Done!")
