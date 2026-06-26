from __future__ import annotations

import math, sys 
from pathlib import Path
import torch 
import torch.nn as nn
import torch.nn.functional as F

#miocodec import path
_mp = Path(__file__).resolve().parent.parent / "external" / "MioCodec" / "src"
if str(_mp) not in sys.path:
    sys.path.insert(0, str(_mp))

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps 
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mena(-1, keepdim=True) + self.eps) * self.weight
    

class CausalMambaBlock(nn.Module):

        def __init__(self, dim: int, cond_dim: int, d_state: int=16,
                    expand: int = 2, d_conv: int = 2, dt_rank: int | str = 'auto'):
        super().__init__()
        from miocodec.module.adaln_zero import AdaLNZero

        self.dim = dim
        self.d_inner = int(expand * dim)
        self.d_state = d_state
        self.d_conv = d_conv

        if dt_rank == 'auto':
            dt_rank = max(1, math.ceil(dim / 16))
        self.dt_rank = dt_rank
        #adanln-zero zeroinit speaker condi at input
        self.adaln = AdaLNZero(dim, cond_dim, return_gate=True)
        # inpurt projection
        self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, bias=True,
                                kernel_size=d_conv, groups=self.d_inner,
                                padding=0) #manual causal padding in forward state
    
        self.x_proj = nn.Linear(self.d_inner, dt_rank + d_state * 2, bias=False)

        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)

        # A : ssm state trasition (learned, inpurt-independent)

        A = torch.arage(1, d_state + 1, dtype=torch.float32), unsqueeze(0) # (1, n) 
        self.A_log = nn.Parameter(torch.log(A).repeat(self.d_inner, 1)) #(d_inner, n)

        # D: skip connection param
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:

        residual = x
        if condition.dim() == 2:
            condition = condition.unsqueeze(1)
        
        B, L, D = x.shape

        normed, gate = self.adaln(x, condition) #gate b,l,d normed b,l,d

        #mamba forward
        #input projection
        x2 = self.in_proj(normed)
        x_in, z = xz.chunk(2, dim=-1) 

        x_in = x_in.transpose(1, 2)
        x_in = F.pad(x_in, (self.d_conv -1, 0))
        x_in = self.conv1d(x_in)
        x_in = F.silu(x_in)
        x_in = x_in.traspose(1, 2)

        y = self._ssm(x_in)
        y = y * F.silu(z)

        out = self.out.proj(y)

        return residual + gate * out

    def _ssm(self, x: torch.Tensor) -> torch.Tensor:

        B, L, d_inner = x.shape
        n = self.d_state

        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        x_dbl = self.x_proj(x)
        delta, ssm_B, ssm_C = x_dbl.split([self.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        deltaB_u = delta.unsqueeze(-1) * ssm_B.unsqueeze(-2) * x.unsqueeze(-1)

        state = torch.zeros(B, d_inner, n, device=x.device, dtype=x.dtype)
        outputs = []
        for i in range(L):
            state = deltaA[:, i] * state + deltaB_u[:, i]
            out_i = (state * ssm_C[:, i, :].unsqueeze(-2)).sum(-1)
            outputs.append(out_i)
        y = torch.stack(outputs, dim=1)

        y = y + x * D.unsqueeze(0).unsqueeze(0)
        return y


#test

if __name__ == "__main__":
    B, L, D, Cd = 2, 50, 384, 128
    x = torch.randn(B, L, D)
    cond = torch.randn(B, 1, Cd)

    block = CausalMambaBlock(D, Cd, d_state=16, expand=2, d_conv=2)
    n = sum(p.numel() for p in block.parameters())
    print(f"Parans : {n:,} ({n/1e6:.3f}M)")

    out = block(x, cond)
    print(f"input: {list(x.shape)}")
    print(f"output: {list(out.shape)}")
    assert out.shape == x.shape, "shape mismatch"

    half = x [:, :L//2]
    cond_half = cond[:, :1] 
    out_half = block(half, cond_half)
    diff = (out[:, :L//2] -out_half).abs().max().item()
    print(f"causality diff: {diff:.6f}")
    print("done.")









