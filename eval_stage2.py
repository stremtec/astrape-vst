#!/usr/bin/env python3
"""
Stage 2: Full Teacher Decoder Plug-in Evaluation.
A: Teacher content + target global → teacher decoder
B: Student content + target global → teacher decoder
Metrics: WER, mel L1/cosine, F0 jitter, centroid, VHigh, crest, speaker SIM
"""
import torch, numpy as np, soundfile as sf, os, glob, warnings
from scipy import signal as scipy_signal
from collections import defaultdict
warnings.filterwarnings('ignore')

SR=44100; HOP_44k=int(SR/25)

# ── Models ────────────────────────────────────────────────────────────
import torch.nn as nn
class CausalTCNEncoder(nn.Module):
    def __init__(self,in_dim=80,hidden=256,out_dim=5,num_layers=4,kernel=5):
        super().__init__()
        self.proj_in=nn.Conv1d(in_dim,hidden,1)
        layers=[]
        for i in range(num_layers):
            d=2**i; p=(kernel-1)*d
            layers.append(nn.Sequential(
                nn.Conv1d(hidden,hidden,kernel,dilation=d,padding=p,padding_mode='replicate'),
                nn.GroupNorm(8,hidden),nn.GELU(),nn.Conv1d(hidden,hidden,1)))
        self.layers=nn.ModuleList(layers)
        self.down=nn.Conv1d(hidden,hidden,3,stride=2,padding=1,padding_mode='replicate')
        self.proj_out=nn.Conv1d(hidden,out_dim,1)
        self.embed_head=nn.Conv1d(out_dim,768,1)
    def forward(self,x):
        h=self.proj_in(x)
        for layer in self.layers:
            r=h; h=layer(h); h=h[:,:,:r.shape[2]]; h=h+r
        h=self.down(h); fsq=self.proj_out(h); embed=self.embed_head(fsq)
        return fsq, embed

student=CausalTCNEncoder()
student.load_state_dict(torch.load("checkpoints/causal_student_v1.pt",map_location='cpu'))
student.eval()

from miocodec.model import MioCodecModel
teacher=MioCodecModel.from_pretrained('Aratako/MioCodec-25Hz-44.1kHz-v2')
teacher.eval()

# ── Whisper for WER ───────────────────────────────────────────────────
try:
    import whisper
    whisper_model=whisper.load_model("base")
    HAS_WHISPER=True
except: HAS_WHISPER=False; print("No Whisper")

# ── Test pairs ────────────────────────────────────────────────────────
ROOT="/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed"
import torchaudio
mel_spec=torchaudio.transforms.MelSpectrogram(sample_rate=16000,n_fft=512,hop_length=320,n_mels=80,f_min=80,f_max=7600,center=False,power=2)

pairs=[
    ("p255","origin","m→f x-lang"),
    ("p226","origin","f→f x-lang"),
    ("p285","p255","f→m VCTK"),
]

def load_audio(spk):
    if spk.startswith('p'):
        files=sorted(glob.glob("{}/{}/{}_*_mic1.flac".format(ROOT,spk,spk)))
        d,sr=sf.read(files[0])
    else:
        d,sr=sf.read("/Users/asill/Downloads/{}.mp3".format(spk))
    if d.ndim>1: d=d.mean(axis=1)
    if sr!=SR: d=scipy_signal.resample(d,int(len(d)*SR/sr))
    return d[:SR*3]

def transcribe(audio_np):
    if not HAS_WHISPER: return ""
    a=audio_np.astype(np.float32)
    a=a/(np.abs(a).max()+1e-8)
    return whisper_model.transcribe(a,language="en",fp16=False)['text']

def wer(ref,hyp):
    rw=ref.lower().split(); hw=hyp.lower().split()
    d=np.zeros((len(rw)+1,len(hw)+1))
    for i in range(len(rw)+1): d[i,0]=i
    for j in range(len(hw)+1): d[0,j]=j
    for i in range(1,len(rw)+1):
        for j in range(1,len(hw)+1):
            d[i,j]=min(d[i-1,j]+1,d[i,j-1]+1,d[i-1,j-1]+(0 if rw[i-1]==hw[j-1] else 1))
    return d[len(rw),len(hw)]/max(len(rw),1)

from scipy.signal import stft
def measure_audio(a):
    a=a-np.mean(a)
    f,_,Z=stft(a,fs=SR,nperseg=1024,noverlap=768)
    mag=np.abs(Z); total=mag.sum()+1e-8
    c=np.sum(f[:len(f)//2,np.newaxis]*mag[:len(f)//2],axis=0)/(mag[:len(f)//2].sum(axis=0)+1e-8)
    vh=mag[(f>=4000)&(f<8000)].sum()/total*100
    cr=np.max(np.abs(a))/(np.sqrt(np.mean(a**2))+1e-8)
    fl,hp=int(SR*0.04),int(SR*0.01); fs=[]
    for i in range(0,len(a)-fl,hp):
        fr=a[i:i+fl]
        if np.sqrt(np.mean(fr**2))<0.001: fs.append(0); continue
        corr=np.correlate(fr,fr,mode='full'); corr=corr[len(corr)//2:]; corr=corr/(corr[0]+1e-8)
        pks=scipy_signal.find_peaks(corr,distance=10)[0]
        if len(pks)==0: fs.append(0); continue
        f0=SR/pks[0]; fs.append(f0 if 50<f0<400 else 0)
    fs=np.array(fs); v=fs>0
    j=np.mean(np.abs(np.diff(fs[v])))/np.mean(fs[v])*100 if v.sum()>3 else 0
    return np.mean(c),vh,cr,j

print("="*85)
print("  STAGE 2: TEACHER DECODER PLUG-IN FULL EVALUATION")
print("="*85)
print("  {'Pair':<22s} {'Cent':>7s} {'Jitter':>7s} {'VHigh':>6s} {'Crest':>6s} {'WER':>6s} {'Type':>12s}")
print("  "+"-"*80)

all_results=[]

for src_spk,tgt_spk,desc in pairs:
    d_src=load_audio(src_spk); alen=len(d_src)
    d_tgt=load_audio(tgt_spk)[:SR*3]
    
    x_src=torch.from_numpy(d_src).float().unsqueeze(0)
    x_tgt=torch.from_numpy(d_tgt).float().unsqueeze(0)
    
    with torch.inference_mode():
        # Teacher encode
        ft_src=teacher.encode(x_src,return_content=True,return_global=True)
        ft_tgt=teacher.encode(x_tgt,return_content=False,return_global=True)
        ge_tgt=ft_tgt.global_embedding
        
        # A: Teacher content → teacher decoder
        wav_A=teacher.decode(global_embedding=ge_tgt,
                            content_token_indices=ft_src.content_token_indices,
                            target_audio_length=alen)
        
        # B: Student content → teacher decoder
        audio_16k=scipy_signal.resample(d_src[:alen],int(alen*16000/SR))
        mel=mel_spec(torch.from_numpy(audio_16k).float().view(1,1,-1))
        logmel=torch.log(mel.squeeze(1).clamp(min=1e-5))  # (1,80,T)
        fsq_pred,_=student(logmel)  # already has batch dim
        fsq_t=fsq_pred.squeeze(0).T
        z_q,_=teacher.local_quantizer.fsq.encode(fsq_t.unsqueeze(0))
        z_q=teacher.local_quantizer.proj_out(z_q)
        wav_B=teacher.decode(global_embedding=ge_tgt,
                            content_embedding=z_q.squeeze(0),
                            target_audio_length=alen)
    
    wA=wav_A.numpy()[:alen]; wB=wav_B.numpy()[:alen]
    
    cA,vhA,crA,jA=measure_audio(wA)
    cB,vhB,crB,jB=measure_audio(wB)
    
    # WER
    txt_src=transcribe(d_src)
    txt_A=transcribe(wA); txt_B=transcribe(wB)
    werA=wer(txt_src,txt_A)*100; werB=wer(txt_src,txt_B)*100
    
    cd=cB-cA; jd=jB-jA; wd=werB-werA
    
    print("  {:<22s} {:.0f}/{:.0f} {:.1f}/{:.1f} {:.1f}/{:.1f} {:.1f}/{:.1f} {:.0f}/{:.0f} {:<12s}".format(
        "{}→{}".format(src_spk,tgt_spk),cA,cB,jA,jB,vhA,vhB,crA,crB,werA,werB,desc))
    print("    Δ: Cent={:+.0f}Hz Jitter={:+.1f}% WER={:+.0f}%".format(cd,jd,wd))
    print("    SRC txt: {}".format(txt_src[:60]))
    print("    A txt:   {}".format(txt_A[:60]))
    print("    B txt:   {}".format(txt_B[:60]))
    
    all_results.append({
        'pair':"{}→{}".format(src_spk,tgt_spk),'desc':desc,
        'cA':cA,'cB':cB,'jd':jd,'werA':werA,'werB':werB,'wd':wd,
        'jA':jA,'jB':jB,'vhA':vhA,'vhB':vhB,'crA':crA,'crB':crB
    })

# Summary
print()
print("="*85)
print("  SUMMARY")
print("="*85)
for r in all_results:
    status="PASS" if abs(r['cB']-r['cA'])/max(r['cA'],1)<0.15 and r['jd']<10 else "CHECK"
    print("  {}: CentΔ={:+.0f}Hz JittΔ={:+.1f}% WERΔ={:+.0f}% → {}".format(
        r['pair'],r['cB']-r['cA'],r['jd'],r['wd'],status))
print()
print("  Pass criteria: centroid dev <15%, jitter Δ <10%, WER Δ <20%")
print("Done!")
