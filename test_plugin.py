#!/usr/bin/env python3
"""Quick plug-in test using saved checkpoint."""
import torch, numpy as np, soundfile as sf
from scipy import signal as scipy_signal
from miocodec.model import MioCodecModel
from train_student_v1_fast import CausalTCNEncoder
import torchaudio

SR=44100

teacher=MioCodecModel.from_pretrained('Aratako/MioCodec-25Hz-44.1kHz-v2'); teacher.eval()
model=CausalTCNEncoder(); model.load_state_dict(torch.load('checkpoints/causal_student_v1.pt',map_location='cpu')); model.eval()

d,sr=sf.read('/Users/asill/asill/research2/datasets/vctk/wav48_silence_trimmed/p255/p255_001_mic1.flac')
if d.ndim>1: d=d.mean(axis=1)
if sr!=SR: d=scipy_signal.resample(d,int(len(d)*SR/sr))
d=d[:SR*3]; alen=len(d)

x_t=torch.from_numpy(d).float().unsqueeze(0)
with torch.inference_mode():
    ft=teacher.encode(x_t,return_content=True,return_global=True)
    ge=ft.global_embedding
    wav_t=teacher.decode(global_embedding=ge,content_token_indices=ft.content_token_indices,target_audio_length=alen)

mel_spec=torchaudio.transforms.MelSpectrogram(sample_rate=16000,n_fft=512,hop_length=320,n_mels=80,f_min=80,f_max=7600,center=False,power=2)
audio_16k=scipy_signal.resample(d[:alen],int(alen*16000/SR))
mel=mel_spec(torch.from_numpy(audio_16k).float().view(1,1,-1)).squeeze(1)
logmel=torch.log(mel.clamp(min=1e-5))

with torch.inference_mode():
    fsq_pred,_=model(logmel)
    fsq_t=fsq_pred.squeeze(0).T
    st=teacher.local_quantizer.fsq.codes_to_indices(fsq_t.unsqueeze(0)).squeeze(0).numpy()
    z_q,_=teacher.local_quantizer.fsq.encode(fsq_t.unsqueeze(0))
    z_q=teacher.local_quantizer.proj_out(z_q)
    wav_s=teacher.decode(global_embedding=ge,content_embedding=z_q.squeeze(0),target_audio_length=alen)

from scipy.signal import stft
def m(a):
    a=a-np.mean(a)
    f,_,Z=stft(a,fs=SR,nperseg=1024,noverlap=768)
    mag=np.abs(Z); total=mag.sum()+1e-8
    c=np.sum(f[:len(f)//2,np.newaxis]*mag[:len(f)//2],axis=0)/(mag[:len(f)//2].sum(axis=0)+1e-8)
    return np.mean(c)

wt=wav_t.numpy()[:alen]; ws=wav_s.numpy()[:alen]
ct=m(wt); cs=m(ws)
print('Teacher: {:.0f}Hz | Student: {:.0f}Hz | Delta: {:.0f}Hz'.format(ct,cs,cs-ct))
tt=ft.content_token_indices.numpy()
match=(tt[:len(st)]==st[:len(tt)]).mean()*100
print('Token match: {:.1f}%'.format(match))
sf.write('/Users/asill/Desktop/mio_student_v1.wav',ws,SR)
print('Done!')
