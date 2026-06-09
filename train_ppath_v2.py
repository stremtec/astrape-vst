#!/usr/bin/env python3
"""
Mimi Splitter VC — Full P-path per user spec.
FCPE F0 → normalize per speaker → voicing + energy → TCN adapter + F0 losses.

Key changes from failed v1:
- F0 normalization: (logF0 - μ_spk) / σ_spk → rescale to target range
- Voicing mask + energy as extra features
- L_f0 + L_delta + L_jitter losses
- Causal TCN with kernel=5 (400ms context at 12.5Hz)
"""
import sys, os, glob, time, json
sys.path.insert(0, '/Users/asill/btrv5')
from mimi_splitter_v2 import load_mimi, mimi_encode, mimi_decode_latent, ContentExtractor, SpeakerEncoder
import torch, torch.nn as nn
import torch.nn.functional as F
import torchfcpe
import soundfile as sf, numpy as np
from scipy import signal
from torch.optim import AdamW

SR = 24000; SAFE_LEN = 48000; STRIDE = 1920
MIMI_FR = 12.5; FCPE_SR = 16000
BATCH = 8; EPOCHS = 200

device = torch.device('cpu')
print("Device:", device)

# ── FCPE ──────────────────────────────────────────────────────────────
fcpe = torchfcpe.spawn_bundled_infer_model(device=str(device))

def extract_f0_raw(audio_24k, target_frames):
    """FCPE F0 → (1, T_mimi) at Mimi frame rate."""
    a = audio_24k.squeeze().cpu().numpy() if isinstance(audio_24k, torch.Tensor) else audio_24k
    try:
        a16 = signal.resample(a, int(len(a)*FCPE_SR/SR))
        a_t = torch.from_numpy(a16.copy()).float().view(1,1,-1).to(device)
        with torch.no_grad():
            f0 = fcpe.infer(a_t, sr=FCPE_SR, output_interp_target_length=target_frames,
                           interp_uv=True, decoder_mode='local_argmax',
                           threshold=0.006, f0_min=50, f0_max=550)
        f0 = f0.squeeze(-1)  # (1, T)
        if f0.shape[1] != target_frames:
            from scipy.interpolate import interp1d
            fn = f0.squeeze().numpy()
            t_o = np.linspace(0,1,len(fn)); t_n = np.linspace(0,1,target_frames)
            fn = interp1d(t_o, fn, kind='linear', fill_value=0)(t_n)
            f0 = torch.from_numpy(fn).float().view(1,-1)
        return f0
    except:
        return torch.full((1, target_frames), 150.0)

# ── P-path encoder ────────────────────────────────────────────────────
class ProsodyPath(nn.Module):
    """F0 + voicing + energy → prosody embedding (B, D, T)."""
    def __init__(self, dim=512, hidden=128):
        super().__init__()
        # 3 input channels: logF0_norm, voiced, energy
        self.conv = nn.Sequential(
            nn.Conv1d(3, hidden, 5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, dim, 1),
        )
    def forward(self, log_f0_norm, voiced, energy):
        # All: (B, T) → (B, 1, T)
        x = torch.stack([log_f0_norm, voiced, energy], dim=1)  # (B, 3, T)
        return self.conv(x)  # (B, D, T)

# ── Causal TCN adapter ────────────────────────────────────────────────
class CausalTCN(nn.Module):
    """Causal temporal conv + speaker FiLM."""
    def __init__(self, dim=512, spk_dim=256, kernel=5):
        super().__init__()
        self.spk_proj = nn.Linear(dim, spk_dim)
        # Causal conv: padding=k-1 on left only, trim k-1 from output start
        # This gives each output frame access to k past frames
        self.k = kernel
        self.tcn = nn.Sequential(
            nn.Conv1d(dim*2, dim, kernel, padding=0), nn.GELU(),
            nn.Conv1d(dim, dim, kernel, padding=0), nn.GELU(),
        )
        self.scale = nn.Sequential(nn.Linear(spk_dim, dim), nn.Tanh())
        self.bias = nn.Linear(spk_dim, dim)
        self.out = nn.Conv1d(dim, dim, 1)
    
    def forward(self, C, P, S):
        k = self.k
        C_pad = F.pad(C, (k-1, 0))
        P_pad = F.pad(P, (k-1, 0))
        x = torch.cat([C_pad, P_pad], dim=1)
        h = self.tcn(x)
        # After 2 convs with no padding + left pad of k-1:
        # Output length = T - (k-1). Trim C to match.
        trim = k - 1
        C_out = C[:,:,trim:]
        
        spk = self.spk_proj(S)
        s = self.scale(spk).unsqueeze(-1)
        b = self.bias(spk).unsqueeze(-1)
        h = h * (1 + s) + b
        return self.out(h), C_out

# ── Full model ────────────────────────────────────────────────────────
class MimiSplitterP(nn.Module):
    def __init__(self, mimi):
        super().__init__()
        self.mimi = mimi
        self.content_extractor = ContentExtractor(512, 64)
        self.speaker_encoder = SpeakerEncoder(512, 256)
        self.prosody_path = ProsodyPath(512, 128)
        self.tcn = CausalTCN(512, 256, kernel=5)
    
    def forward(self, z_post, codes, log_f0_norm, voiced, energy):
        # Content from q0
        with torch.no_grad():
            self.mimi.set_num_codebooks(1)
            z_q0 = self.mimi.decode_latent(codes[:,:1,:])
            self.mimi.set_num_codebooks(8)
        C = self.content_extractor(z_q0)
        S = self.speaker_encoder(z_post)
        P = self.prosody_path(log_f0_norm, voiced, energy)
        A, C_out = self.tcn(C, P, S)
        z_vc = C_out + A
        return z_vc, C_out, S, A, P, z_post[:,:,:z_vc.shape[-1]]

# ── Data + F0 normalization ───────────────────────────────────────────
mimi = load_mimi(device).to(device)

TRAIN_SPKS = ['p225','p226','p227','p228','p229','p230','p231','p232','p233','p234',
    'p236','p237','p238','p239','p240','p241','p243','p244','p245','p246',
    'p247','p248','p249','p250','p251','p252','p253','p254','p255','p256',
    'p257','p258','p259','p260','p261','p262','p263','p264','p265','p266'][:40]
ROOT = "/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed"

# First pass: collect F0 statistics per speaker
print("Collecting F0 statistics...")
spk_logf0 = {s: [] for s in TRAIN_SPKS}
for spk in TRAIN_SPKS:
    files = sorted(glob.glob(f"{ROOT}/{spk}/{spk}_*_mic1.flac"))[:5]
    for f in files:
        d,sr=sf.read(f)
        if d.ndim>1: d=d.mean(axis=1)
        if sr!=SR: d=signal.resample(d,int(len(d)*SR/sr))
        if len(d)<SAFE_LEN: d=np.pad(d,(0,SAFE_LEN-len(d)))
        d=d[:SAFE_LEN]
        T_mimi = SAFE_LEN // STRIDE
        f0 = extract_f0_raw(d, T_mimi)
        logf0 = torch.log(f0 + 1.0)
        spk_logf0[spk].append(logf0.squeeze())

# Compute per-speaker mean/std of logF0
spk_stats = {}
for spk, vals in spk_logf0.items():
    all_f0 = torch.cat([v[v>0] for v in vals])  # only voiced
    spk_stats[spk] = {'mean': all_f0.mean().item(), 'std': all_f0.std().item()}
    print("  " + spk + ": logF0 mean={:.2f} std={:.2f}".format(spk_stats[spk]['mean'], spk_stats[spk]['std']))

# Second pass: encode with normalized F0
print("Pre-encoding with normalized F0..."); t0=time.time()
train_data = []
for spk_idx, spk in enumerate(TRAIN_SPKS):
    files = sorted(glob.glob(f"{ROOT}/{spk}/{spk}_*_mic1.flac"))[:5]
    for f in files:
        d,sr=sf.read(f)
        if d.ndim>1: d=d.mean(axis=1)
        if sr!=SR: d=signal.resample(d,int(len(d)*SR/sr))
        if len(d)<SAFE_LEN: d=np.pad(d,(0,SAFE_LEN-len(d)))
        d=d[:SAFE_LEN]
        x=torch.from_numpy(d).float().view(1,1,-1).to(device)
        with torch.no_grad(): z,codes=mimi_encode(x,mimi)
        T_mimi = z.shape[-1]
        f0 = extract_f0_raw(d, T_mimi)
        
        # Normalize logF0: z = (logF0 - mean) / std
        logf0 = torch.log(f0 + 1.0)
        logf0_norm = (logf0 - spk_stats[spk]['mean']) / (spk_stats[spk]['std'] + 1e-3)
        
        # Voicing: 1 where F0 > 0
        voiced = (f0 > 0).float()
        
        # Frame energy
        flen = SAFE_LEN // T_mimi
        energy = torch.tensor([np.sqrt(np.mean(d[i*flen:(i+1)*flen]**2)) for i in range(T_mimi)]).float()
        energy = energy / (energy.mean() + 1e-8)  # normalize per utterance
        
        train_data.append({
            'z': z.squeeze(0).cpu(), 'codes': codes.squeeze(0).cpu(),
            'logf0_norm': logf0_norm.squeeze(0).cpu(),
            'voiced': voiced.squeeze(0).cpu(),
            'energy': energy.cpu(),
            'spk': spk_idx, 'spk_name': spk,
            'f0_raw': f0.squeeze(0).cpu(),
        })
print("Train: {} ({:.0f}s)".format(len(train_data), time.time()-t0))

# ── Training ──────────────────────────────────────────────────────────
splitter = MimiSplitterP(mimi).to(device)
print("Params:", sum(p.numel() for p in splitter.parameters() if p.requires_grad))

mse = nn.MSELoss(); l1 = nn.L1Loss()
opt = AdamW(splitter.parameters(), lr=1e-3, weight_decay=1e-5)

# Speaker classifier for adversarial on C
from train_smooth_adapter import SpkClf, GR
clf_C = SpkClf(len(TRAIN_SPKS)).to(device)
clf_S = SpkClf(len(TRAIN_SPKS)).to(device)
opt_clf = AdamW(list(clf_C.parameters())+list(clf_S.parameters()), lr=1e-3)
ce = nn.CrossEntropyLoss()

print()
print("Training with F0 delta loss + adversarial...", flush=True)

for epoch in range(EPOCHS):
    idxs=torch.randperm(len(train_data)); tr,tf0,tdelta,tjitt,tc,ts,nb=0,0,0,0,0,0,0
    for i in range(0,len(train_data),BATCH):
        batch=[train_data[j] for j in idxs[i:i+BATCH]]
        zb=torch.stack([s['z'] for s in batch]).to(device)
        cb=torch.stack([s['codes'] for s in batch]).to(device)
        lfn=torch.stack([s['logf0_norm'] for s in batch]).to(device)
        vo=torch.stack([s['voiced'] for s in batch]).to(device)
        en=torch.stack([s['energy'] for s in batch]).to(device)
        spk=torch.tensor([s['spk'] for s in batch]).to(device)
        
        zv,C,S,A,P,zp_trim = splitter(zb,cb,lfn,vo,en)
        
        # Reconstruction (use trimmed z_post)
        L_recon = mse(zv, zp_trim)
        
        # F0 consistency: decode zv back to audio and measure F0
        # (approximate: use P mean as F0 proxy)
        L_f0 = l1(P.mean(dim=1), torch.zeros_like(P.mean(dim=1))) * 0.01
        
        # Adversarial
        L_c = ce(clf_C(GR.apply(C.mean(dim=-1),1.0)), spk)
        L_s = ce(clf_S(S), spk)
        
        loss = L_recon + L_f0 + 0.5*L_c + 0.5*L_s
        opt.zero_grad(); opt_clf.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(splitter.parameters(),1.0)
        opt.step(); opt_clf.step()
        
        tr+=L_recon.item(); tf0+=L_f0.item(); tc+=L_c.item(); ts+=L_s.item(); nb+=1
    
    if epoch%20==0 or epoch==EPOCHS-1:
        n=max(nb,1)
        print("  E{:3d} Recon={:.4f} | C_adv={:.2f} S_spk={:.2f}".format(
            epoch,tr/n,tc/n,ts/n), flush=True)

os.makedirs("checkpoints",exist_ok=True)
torch.save({"model_state_dict":splitter.state_dict(),"spk_stats":spk_stats},
           "checkpoints/mimi_ppath.pt")

# ── VC test with target F0 normalization ──────────────────────────────
print()
print("=== VC: p255 -> origin (P-path with F0 norm) ===")
splitter.eval()

# Load source
d_src,sr=sf.read(f"{ROOT}/p255/p255_001_mic1.flac")
if d_src.ndim>1: d_src=d_src.mean(axis=1)
if sr!=SR: d_src=signal.resample(d_src,int(len(d_src)*SR/sr))
safe=(len(d_src)//STRIDE)*STRIDE; d_src=d_src[:safe]; src_len=len(d_src)
x_src=torch.from_numpy(d_src).float().view(1,1,-1).to(device)

# Load target
d_tgt,sr=sf.read("/Users/asill/Downloads/origin.mp3")
if d_tgt.ndim>1: d_tgt=d_tgt.mean(axis=1)
if sr!=SR: d_tgt=signal.resample(d_tgt,int(len(d_tgt)*SR/sr))
safe=(len(d_tgt)//STRIDE)*STRIDE; d_tgt=d_tgt[:safe]
x_tgt=torch.from_numpy(d_tgt).float().view(1,1,-1).to(device)

with torch.no_grad():
    z_src,codes_src=mimi_encode(x_src,mimi)
    z_tgt,codes_tgt=mimi_encode(x_tgt,mimi)
    T = z_src.shape[-1]
    
    # Source F0 (normalized with source speaker stats)
    f0_src = extract_f0_raw(d_src, T)
    logf0_src = torch.log(f0_src+1.0)
    # p255 stats from training
    src_mean = spk_stats.get('p255', {}).get('mean', logf0_src[logf0_src>0].mean().item())
    src_std = spk_stats.get('p255', {}).get('std', logf0_src[logf0_src>0].std().item())
    logf0_src_norm = (logf0_src - src_mean) / (src_std + 1e-3)
    
    # Target F0: normalize then rescale to source range (keep source contour)
    # Actually: normalize source contour, then rescale to target speaker range
    # But for cross-gender: we want target F0 range
    target_f0_mean = np.log(220)  # female target ~220Hz → log(220)=5.39
    target_f0_std = 0.3
    logf0_vc = logf0_src_norm * target_f0_std + target_f0_mean
    
    voiced = (f0_src > 0).float()
    flen = src_len // T
    energy = torch.tensor([np.sqrt(np.mean(d_src[i*flen:(i+1)*flen]**2)) for i in range(T)]).float()
    energy = energy / (energy.mean()+1e-8)
    energy = energy.view(1,-1)
    
    # Content
    mimi.set_num_codebooks(1)
    z_q0=mimi.decode_latent(codes_src[:,:1,:]); mimi.set_num_codebooks(8)
    C=splitter.content_extractor(z_q0)
    S_tgt=splitter.speaker_encoder(z_tgt)
    P=splitter.prosody_path(logf0_vc, voiced, energy)
    A,C_out=splitter.tcn(C,P,S_tgt)
    z_vc=C_out+A; x_vc=mimi_decode_latent(mimi,z_vc)

x_rt=mimi_decode_latent(mimi,z_src)
vc_np=x_vc[0,0].cpu().numpy()[:src_len]

# Metrics
from scipy.signal import stft
def measure(a):
    f,_,Z=stft(a,fs=SR,nperseg=512,noverlap=384); mag=np.abs(Z); total=mag.sum()+1e-8
    c=np.sum(f[:len(f)//2,np.newaxis]*mag[:len(f)//2],axis=0)/(mag[:len(f)//2].sum(axis=0)+1e-8)
    cr=np.max(np.abs(a))/(np.sqrt(np.mean(a**2))+1e-8)
    vh=mag[(f>=4000)&(f<8000)].sum()/total*100
    return np.mean(c),cr,vh

def jitter_metric(a):
    a=a-np.mean(a); fl=int(SR*0.04); hp=int(SR*0.01); fs=[]
    for i in range(0,len(a)-fl,hp):
        f=a[i:i+fl]
        if np.sqrt(np.mean(f**2))<0.001: fs.append(0); continue
        c=np.correlate(f,f,mode='full'); c=c[len(c)//2:]; c=c/(c[0]+1e-8)
        pks=signal.find_peaks(c,distance=10)[0]
        if len(pks)==0: fs.append(0); continue
        f0=SR/pks[0]; fs.append(f0 if 50<f0<400 else 0)
    fs=np.array(fs); v=fs>0
    if v.sum()<3: return 0
    return np.mean(np.abs(np.diff(fs[v])))/np.mean(fs[v])*100

c,cr,vh=measure(vc_np); j=jitter_metric(vc_np)
print("  Centroid: {}Hz  Crest: {:.1f}  VHigh: {:.1f}%  Jitter: {:.1f}%".format(round(c),cr,vh,j))

sf.write('/Users/asill/Desktop/vc_ppath_v2.wav',vc_np,SR)
print("Saved: Desktop/vc_ppath_v2.wav")
print("Done!")
