"""Fine-tune center=True model on center=False mel."""
import sys,warnings,logging,random,time,json,argparse
warnings.filterwarnings("ignore");logging.disable(logging.INFO)
sys.path.insert(0,"external/MioCodec/src");sys.path.insert(0,".")
import torch,torchaudio,numpy as np,torch.nn as nn,torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset,DataLoader
from mcs_common import *
from mcs_q2d2 import compute_q2d2_perplexity
from train_mcs_q2d2 import MCSTransQ2D2Config,MCSTransQ2D2,q2d2_losses
from eval_mcs_trans_audio import load_mio,load_wave,SAMPLE_RATE
S=SAMPLE_RATE;HOP=882;NFFT=2048

class D(Dataset):
    def __init__(self,idxs,spks,srcs,mx=3.0):
        self.idx=[int(i)for i in idxs];self.spk=spks;self.src=srcs
        self.ms=int(mx*S);self.rng=random.Random(42)
    def __len__(self):return len(self.idx)
    def __getitem__(self,i):
        import soundfile as sf
        ii=self.idx[i];d,sr=sf.read(str(Path(self.src[ii])),dtype="float32")
        d=torch.from_numpy(np.asarray(d))
        if d.ndim==2:d=d.mean(1)
        if sr!=S:d=torchaudio.functional.resample(d.unsqueeze(0),sr,S).squeeze(0)
        if d.shape[0]>self.ms:s=self.rng.randint(0,d.shape[0]-self.ms);d=d[s:s+self.ms]
        elif d.shape[0]<self.ms:d=F.pad(d,(0,self.ms-d.shape[0]))
        m=torchaudio.transforms.MelSpectrogram(sample_rate=S,n_fft=NFFT,hop_length=HOP,n_mels=80,f_min=0.0,f_max=S/2.0,power=1,center=False)(d.unsqueeze(0))
        m=torch.log(torch.clamp(m,min=1e-5))
        c=torch.from_numpy(np.load(Path("data/mio_vctk_full_compact")/f"s_{ii:05d}.npz",allow_pickle=False)["ce_768"].astype(np.float32))
        return m[0].float(),c.float(),str(self.spk[ii]),ii,str(self.src[ii])

def coll(smps,ms=6.0):
    B=len(smps);MF=int(ms*50);CF=int(ms*25)
    ml=torch.zeros(B,80,MF);ct=[];mk=torch.zeros(B,CF,dtype=torch.bool)
    ss=[];ii=[];sr=[]
    for i,(m,c,sp,idx,src) in enumerate(smps):
        mf=min(m.shape[1],MF);ml[i,:,:mf]=m[:,:mf]
        cf=min(c.shape[0],CF)
        if c.shape[0]<CF:c=F.pad(c,(0,0,0,CF-c.shape[0]))
        ct.append(c[:CF]);mk[i,:cf]=True;ss.append(sp);ii.append(idx);sr.append(src)
    return Batch(mel=ml,content=torch.stack(ct),tokens=torch.zeros(B,CF,dtype=torch.long),mask=mk,speakers=ss,indices=torch.tensor(ii,dtype=torch.long),crop_starts=torch.zeros(B,dtype=torch.long)),sr

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dev",default="mps");ap.add_argument("--ep",type=int,default=10)
    ap.add_argument("--sp",type=int,default=2000);ap.add_argument("--bs",type=int,default=2)
    ap.add_argument("--ms",type=float,default=6.0);ap.add_argument("--lr",type=float,default=5e-5)
    ap.add_argument("--dww",type=float,default=1.0);ap.add_argument("--dwp",type=float,default=0.3)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--out",type=Path,default=Path("/Volumes/UNTITLED/btrv5_checkpoints/cf_finetune"))
    ap.add_argument("--init",type=Path,default=Path("/Volumes/UNTITLED/btrv5_checkpoints/mcs_trans_q2d2_grl/mcs_trans_q2d2_grl.best.pt"))
    args=ap.parse_args()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    dev=torch.device(args.dev)
    meta=np.load(DEFAULT_DATA_DIR/"meta.npz",allow_pickle=False)
    n=int(meta["n_samples"]);spk=meta["spk_names"][:n].astype(str);src=meta["source_files"][:n].astype(str)
    ti,vi=split_by_speaker(spk,0.05,args.seed);pi=speaker_balanced_subset(vi,spk,256,args.seed)
    td=D(ti,spk,src,args.ms);pd=D(pi,spk,src,args.ms)
    tl=DataLoader(td,args.bs,shuffle=True,collate_fn=lambda x:coll(x,args.ms))
    pl=DataLoader(pd,args.bs,shuffle=False,collate_fn=lambda x:coll(x,args.ms))
    ck=torch.load(args.init,map_location="cpu",weights_only=False);cfg=ck.get("config",{})
    cfg2={k:(tuple(v) if isinstance(v,list) else v) for k,v in cfg.items() if not k.startswith("_")}; cfg2["grl_weight"]=0.0; cfg2["grl_num_speakers"]=0; config=MCSTransQ2D2Config(**cfg2)
    model=MCSTransQ2D2(config).to(dev)
    model.load_state_dict(ck["state_dict"],strict=False)
    print(f"Init OK, {sum(p.numel() for p in model.parameters()):,}p",flush=True)
    mio=load_mio("cpu").eval();[setattr(p,"requires_grad",False) for p in mio.parameters()]
    nffts=(512,1024,2048);src_all=src
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.ep)
    args.out.mkdir(parents=True,exist_ok=True);q=model.q2d2.quantizer
    best_cos=-1.0;t0=time.time()
    ns=argparse.Namespace(content_cos_weight=1.0,content_l1_weight=0.5,delta_weight=0.04,voiced_boost=1.0,grl_weight=0.0)
    print(f"cf ft: dw={args.dww} p={args.dwp}",flush=True)
    for ep in range(args.ep):
        model.train();tot={}
        for st,(batch,_) in enumerate(tl,1):
            if st>args.sp:break
            batch=move_batch(batch,dev);out=model(batch.mel,batch.mask)
            cl,metrics=q2d2_losses(out,batch,ns,q)
            wl=torch.tensor(0.0,device=dev)
            if random.random()<args.dwp:
                ii=random.randrange(len(batch.speakers))
                row=int(batch.indices[ii].item());sp=Path(str(src_all[row]))
                if sp.exists():
                    ow_tmp=load_wave(sp,SAMPLE_RATE,max_seconds=args.ms);ow=ow_tmp
                    mcs=int(batch.crop_starts[ii].item());ws=mcs*882
                    wlen=int(args.ms*25*1764);ow=ow[ws:ws+wlen]
                    with torch.no_grad():fe=mio.encode(ow_tmp.unsqueeze(0),return_content=False,return_global=True);sl=mio._calculate_target_stft_length(ow.numel())
                    ci=out["projected"][ii].unsqueeze(0).transpose(1,2)
                    nf=min(ci.shape[1],99)
                    pw=mio.forward_wave(ci.cpu()[:,:nf],fe.global_embedding.unsqueeze(0),stft_length=sl).squeeze(0)
                    tl_=min(pw.shape[-1],ow.shape[-1]);wl=multi_resolution_stft_loss(pw[:tl_],ow[:tl_],nffts)
            loss=cl+args.dww*wl;opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            for k,v in metrics.items():tot[k]=tot.get(k,0)+v
            usage=metrics.get("q2d2_usage",0)
            l1=metrics.get("l1",0)
            tot["loss"]=tot.get("loss",0)+float(loss.cpu());tot["wave"]=tot.get("wave",0)+float(wl.cpu())
            if st%50==0:
                d=max(st,1)
                us=getattr(metrics,'q2d2_usage',0) if 'q2d2_usage' in dir(metrics) else metrics.get('q2d2_usage',0)
                us_avg=tot.get('q2d2_usage',0)/max(st,1) if tot.get('q2d2_usage',0)>0 else 0
                l1_avg=tot.get('content_l1',0)/max(st,1)
                print(f"E{ep:03d}s{st:04d} L={tot['loss']/d:.4f} cos={tot['cos768']/d:.4f} w={tot['wave']/d:.4f} u={us_avg:.3f} l1={l1_avg:.4f}",flush=True)
        sch.step()
        model.eval();pb={}
        for batch,_ in pl:
            batch=move_batch(batch,dev);out=model(batch.mel,batch.mask);_,m=q2d2_losses(out,batch,ns,q)
            for k,v in m.items():pb.setdefault(k,[]).append(v)
        model.train();probe={k:float(np.mean(v))for k,v in pb.items()}
        cc=probe.get("cos768",0);print(f"E{ep:03d} probe cos={cc:.4f}",flush=True)
        mf={"epoch":ep,"gs":(ep+1)*args.sp,"probe":probe,"elapsed":time.time()-t0}
        save_checkpoint(args.out/"last.pt",model,opt,sch,ep,mf,args,best_cos)
        if cc>best_cos:best_cos=cc;save_checkpoint(args.out/"best.pt",model,opt,sch,ep,mf,args,best_cos)
        (args.out/"summary.json").write_text(json.dumps(mf,indent=2)+"\n")
    print(f"done best={best_cos:.4f}",flush=True)
if __name__=="__main__":main()
