"""
Q2D2: Two-Dimensional Quantization for MCS-Trans Encoder.

Faithful implementation of the ICML 2026 paper:
"Two-Dimensional Quantization for Geometry-Aware Audio Coding"
by Tal Shuster, Eliya Nachmani.

Reference: https://arxiv.org/abs/2512.01537
Official code: https://github.com/tashQ/Q2D2

Integrated as a drop-in replacement for FSQ in MCS-Trans.
Allows MioCodec decoder compatibility via learned projection out.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ─────────────────────────────────────────────
# 1. Grid Generation (Algorithms 1–3 from paper)
# ─────────────────────────────────────────────


def generate_hex_grid(
    levels: int,
    device: torch.device,
    extent: float | None = None,
) -> Tensor:
    """
    Algorithm 1: Hexagonal tiling grid.
    
    Uses alternating row offsets for hexagonal packing.
    
    Args:
        levels: Number of grid steps along each axis (≥ 2).
        extent: Range symmetric around 0. Default: (levels-1)/2.
    
    Returns:
        grid: (G, 2) tensor of hexagonal grid coordinates.
    """
    assert levels >= 2, "levels must be >= 2"
    if extent is None:
        extent = (levels - 1) / 2.0
    
    dx = 2.0 * extent / (levels - 1)
    dy = dx * (3.0 ** 0.5) / 2.0
    
    y_coords = torch.linspace(-extent, extent, levels, device=device)
    grid_points: list[Tensor] = []
    
    for i, y in enumerate(y_coords):
        x_offset = (-dx / 4.0) if i % 2 else (dx / 4.0)
        x_coords = torch.linspace(-extent, extent, levels, device=device) + x_offset
        g = torch.stack(
            torch.meshgrid(x_coords, y[None], indexing="ij"), dim=-1
        ).reshape(-1, 2)
        grid_points.append(g)
    
    return torch.cat(grid_points, dim=0)  # (G, 2)


def generate_rect_grid(
    x_levels: int,
    y_levels: int,
    device: torch.device,
    x_extent: float | None = None,
    y_extent: float | None = None,
) -> Tensor:
    """
    Algorithm 2: Standard rectangular grid.
    
    Args:
        x_levels, y_levels: Number of points along each axis (≥ 2).
    
    Returns:
        grid: (x_levels * y_levels, 2) tensor.
    """
    assert x_levels >= 2 and y_levels >= 2
    if x_extent is None:
        x_extent = (x_levels - 1) / 2.0
    if y_extent is None:
        y_extent = (y_levels - 1) / 2.0
    
    x_coords = torch.linspace(-x_extent, x_extent, x_levels, device=device)
    y_coords = torch.linspace(-y_extent, y_extent, y_levels, device=device)
    grid = torch.stack(
        torch.meshgrid(x_coords, y_coords, indexing="ij"), dim=-1
    ).reshape(-1, 2)
    return grid


def generate_rhombic_grid(
    x_levels: int,
    y_levels: int,
    device: torch.device,
    x_extent: float | None = None,
    y_extent: float | None = None,
) -> Tensor:
    """
    Algorithm 3: Rhombic tiling grid.
    
    Combines regular grid points with rectangle midpoints,
    producing a rhombic pattern with ~2x the point density
    of a rectangular grid at the same level count.
    
    Requires odd levels in both axes to include (0, 0).
    
    Returns:
        grid: Tensor of rhombic lattice coordinates.
    """
    assert x_levels % 2 == 1 and y_levels % 2 == 1, \
        f"Rhombic grid requires odd levels, got ({x_levels}, {y_levels})"
    if x_extent is None:
        x_extent = (x_levels - 1) / 2.0
    if y_extent is None:
        y_extent = (y_levels - 1) / 2.0
    
    dx = 2.0 * x_extent / (x_levels - 1)
    dy = 2.0 * y_extent / (y_levels - 1)
    
    x_coords = torch.linspace(-x_extent, x_extent, x_levels, device=device)
    y_coords = torch.linspace(-y_extent, y_extent, y_levels, device=device)
    
    # regular grid points
    xg, yg = torch.meshgrid(x_coords, y_coords, indexing="ij")
    regular = torch.stack((xg, yg), dim=-1).reshape(-1, 2)
    
    # midpoints of each rectangle
    mid_x = x_coords[:-1] + dx / 2.0
    mid_y = y_coords[:-1] + dy / 2.0
    mx, my = torch.meshgrid(mid_x, mid_y, indexing="ij")
    midpoints = torch.stack((mx, my), dim=-1).reshape(-1, 2)
    
    return torch.cat([regular, midpoints], dim=0)


# ─────────────────────────────────────────────
# 2. Straight-Through Estimator (STE) helpers
# ─────────────────────────────────────────────


def round_ste(z: Tensor) -> Tensor:
    """Round with straight-through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


def ste(z: Tensor, bounded_z: Tensor) -> Tensor:
    """General straight-through estimator."""
    return z + (bounded_z - z).detach()


# ─────────────────────────────────────────────
# 3. Q2D2 Core Quantizer
# ─────────────────────────────────────────────


class Q2D2Quantizer(nn.Module):
    """
    Two-Dimensional Quantization.
    
    Groups feature dimensions into pairs and jointly quantizes
    each pair to the nearest point on a structured 2D grid.
    
    No learned codebook — the grid is fixed and analytic.
    
    Pipeline (from paper §3.1):
        z               (B, T, D)
          → project_in → Linear(D → d) → Tanh  → [-1, 1]^d
          → bound       → z' = z * (levels-1)/2  → [-l_i/2, l_i/2]
          → pair        → reshape to (B, T, d/2, 2)
          → grid snap   → nearest 2D grid point per pair (STE)
          → project_out → Linear(d → D)
          → z_hat       (B, T, D)
    
    Args:
        dim: Input/output feature dimension.
        levels: Per-dimension quantization levels (must be even length).
        vq_type: Grid type: "hexagon", "rectangle", "rhombic".
        noise_dropout: Training-only noise injection for robustness.
    """
    
    def __init__(
        self,
        dim: int,
        levels: List[int],
        vq_type: str = "rhombic",
        noise_dropout: float = 0.0,
        gumbel_temperature: float = 0.0,
        projection_bias: bool = True,
        use_l2_norm: bool = False,
    ):
        super().__init__()
        
        codebook_dim = len(levels)
        assert codebook_dim % 2 == 0, \
            f"Q2D2 requires even dimension, got {codebook_dim}"
        
        self.dim = dim
        self.codebook_dim = codebook_dim
        self.num_pairs = codebook_dim // 2
        self.vq_type = vq_type
        self.noise_dropout = noise_dropout
        self.gumbel_temperature = gumbel_temperature
        self.use_l2_norm = use_l2_norm
        
        # ── learnable projections ──
        self.project_in = nn.Sequential(
            nn.Linear(dim, codebook_dim, bias=projection_bias),
            nn.Tanh(),  # bounds to [-1, 1] before grid alignment
        )
        self.project_out = (
            nn.Linear(codebook_dim, dim, bias=projection_bias)
            if dim != codebook_dim
            else nn.Identity()
        )
        
        # ── level buffer ──
        _levels = torch.tensor(levels, dtype=torch.int32)
        self.register_buffer("_levels", _levels, persistent=False)
        
        # ── build per-pair grids ──
        grids, grid_lens = self._build_grids()
        self.tile_grid = grids  # list of (G_i, 2) tensors
        self.register_buffer(
            "grid_len",
            torch.tensor(grid_lens, dtype=torch.long),
            persistent=False,
        )
        
        # ── basis for index ↔ codes mapping ──
        grid_basis = torch.ones(self.num_pairs, dtype=torch.long)
        for i in range(1, self.num_pairs):
            grid_basis[i] = grid_basis[i - 1] * grid_lens[i - 1]
        self.register_buffer("grid_basis", grid_basis, persistent=False)
        
        # ── precompute half-widths for de-normalization ──
        half_widths = (_levels // 2).float().view(-1, 2)  # (P, 2)
        self.register_buffer("half_widths", half_widths, persistent=False)
    
    # ── grid construction ──
    
    def _build_grids(self) -> Tuple[List[Tensor], List[int]]:
        """Build 2D grids for each feature pair."""
        grids: list[Tensor] = []
        grid_lens: list[int] = []
        device = self._levels.device
        
        for i in range(self.num_pairs):
            if self.vq_type == "hexagon":
                lvl = int(self._levels[2 * i].item())
                extent = (lvl - 1) / 2.0
                g = generate_hex_grid(lvl, device, extent)
            elif self.vq_type == "rectangle":
                xl = int(self._levels[2 * i].item())
                yl = int(self._levels[2 * i + 1].item())
                xe = (xl - 1) / 2.0
                ye = (yl - 1) / 2.0
                g = generate_rect_grid(xl, yl, device, xe, ye)
            elif self.vq_type == "rhombic":
                xl = int(self._levels[2 * i].item())
                yl = int(self._levels[2 * i + 1].item())
                xe = (xl - 1) / 2.0
                ye = (yl - 1) / 2.0
                g = generate_rhombic_grid(xl, yl, device, xe, ye)
            else:
                raise ValueError(f"Unknown vq_type: {self.vq_type}")
            
            grids.append(g)
            grid_lens.append(g.shape[0])
        
        return grids, grid_lens
    
    # ── bounding ──
    
    def _bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        """
        Bound tanh output to the quantization range.
        
        z' = z * (levels_i - 1) * (1 + eps) / 2
        → z'_i ∈ [-(levels_i-1)/2, +(levels_i-1)/2]
        """
        half_l = (self._levels - 1).float() * (1.0 + eps) / 2.0
        return z * half_l
    
    # ── grid snapping ──
    
    def _snap_to_grid(
        self, z_pairs: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Snap each 2D feature pair to its nearest grid point.
        
        Args:
            z_pairs: (..., P, 2) where P = num_pairs.
        
        Returns:
            snapped: (..., P, 2) aligned to grid.
            nearest: (..., P) integer indices per pair.
        """
        device = z_pairs.device
        prefix_shape = z_pairs.shape[:-2]
        P = self.num_pairs
        z_flat = z_pairs.reshape(-1, P, 2)  # (B*T, P, 2)
        
        snapped: list[Tensor] = []
        nearest_all: list[Tensor] = []
        
        for i in range(P):
            g = self.tile_grid[i].to(device)       # (G_i, 2)
            p = z_flat[:, i]                         # (B*T, 2)
            
            # pairwise distances: (B*T, 1) vs (1, G_i) → (B*T, G_i)
            if self.use_l2_norm:
                p = F.normalize(p, p=2, dim=-1)
                gn = F.normalize(g, p=2, dim=-1)
                dists = torch.cdist(p.unsqueeze(1), gn.unsqueeze(0)).squeeze(1)
            else:
                dists = torch.cdist(p.unsqueeze(1), g.unsqueeze(0)).squeeze(1)

            # ── Gumbel-Softmax soft assignment (prefill / exploration) ──
            if self.training and self.gumbel_temperature > 0:
                soft_w = F.gumbel_softmax(-dists, tau=self.gumbel_temperature,
                                          hard=False, dim=-1)        # (B*T, G_i)
                snapped_i = soft_w @ g                                # (B*T, 2)
                n_idx = (-dists).argmax(dim=-1)                        # for util stats
            else:
                # ── hard argmin (original) ──
                n_idx = dists.argmin(dim=-1)
                snapped_i = g[n_idx]                                  # (B*T, 2)

            snapped.append(snapped_i)
            nearest_all.append(n_idx)
        
        snapped = torch.stack(snapped, dim=1)       # (B*T, P, 2)
        nearest = torch.stack(nearest_all, dim=1)   # (B*T, P)
        
        return (
            snapped.view(*prefix_shape, P, 2),
            nearest.view(*prefix_shape, P),
        )
    
    # ── quantization ──
    
    def _quantize(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Full quantization pass.
        
        Steps:
        1. Bound to grid range.
        2. Reshape into pairs.
        3. Snap each pair to nearest grid point (STE gradients).
        4. De-normalize back to [-1, 1] range.
        
        Returns:
            z_codes: Quantized tensor (same shape as z).
            indices: Per-pair grid indices (B, T, num_pairs).
        """
        bounded = self._bound(z)                                          # (B, T, d)
        z_pairs = bounded.reshape(*bounded.shape[:-1], self.num_pairs, 2)  # (B, T, P, 2)
        
        z_snapped, nearest = self._snap_to_grid(z_pairs)                   # (B, T, P, 2)
        
        # optional noise dropout during training
        if self.training and self.noise_dropout > 0.0:
            mask = torch.bernoulli(
                torch.full_like(z_snapped, self.noise_dropout)
            ).bool()
            offset = torch.rand_like(z_snapped) - 0.5
            z_snapped = torch.where(mask, z_snapped + offset, z_snapped)
        
        bounded_q = z_snapped.reshape_as(bounded)  # (B, T, d)

        # De-normalize by half-widths.
        # Gumbel-Softmax: bounded_q is already differentiable, skip STE.
        # Hard snap: use STE to pass gradients through the argmin.
        half_l = (self._levels // 2).float()
        if self.training and self.gumbel_temperature > 0:
            z_codes = bounded_q / half_l    # differentiable, no STE
        else:
            z_codes = ste(z, bounded_q) / half_l   # STE for hard snap
        
        return z_codes, nearest
    
    # ── index ↔ codes ──
    
    def codes_to_indices(self, nearest: Tensor) -> Tensor:
        """Map per-pair grid indices to flattened code."""
        # nearest: (B, T, P)
        indices = (nearest * self.grid_basis).sum(dim=-1).to(torch.int32)
        return indices.unsqueeze(-1)  # (B, T, 1)
    
    def indices_to_codes(self, indices: Tensor) -> Tensor:
        """Reconstruct quantized features from flattened indices."""
        B, T, _ = indices.shape
        device = indices.device
        
        # recover per-pair nearest indices
        nearest = (indices // self.grid_basis) % self.grid_len  # (B, T, P)
        
        z_pairs: list[Tensor] = []
        for i in range(self.num_pairs):
            g = self.tile_grid[i].to(device)              # (G_i, 2)
            ni = nearest[..., i].reshape(-1)              # (B*T,)
            z_pair = g[ni].view(B, T, 2)                 # (B, T, 2)
            z_pairs.append(z_pair)
        
        z = torch.stack(z_pairs, dim=-2)                  # (B, T, P, 2)
        half_l = self.half_widths.to(device)              # (P, 2)
        codes = z / half_l                                 # de-normalize
        
        return codes.reshape(B, T, -1)                    # (B, T, d)
    
    # ── forward ──
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: (B, T, dim) input features.
        
        Returns:
            quantized: (B, T, dim) quantized features.
            indices: (B, T, 1) flattened code indices.
        """
        # project into quantization space
        z = self.project_in(x)                    # (B, T, codebook_dim)
        
        # quantize
        z_q, nearest = self._quantize(z)          # (B, T, codebook_dim), (B, T, P)
        
        # flat indices
        indices = self.codes_to_indices(nearest)  # (B, T, 1)
        
        # project back
        out = self.project_out(z_q)               # (B, T, dim)
        
        return out, indices
    
    def forward_with_nearest(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Like forward() but also returns per-pair nearest indices."""
        z = self.project_in(x)
        z_q, nearest = self._quantize(z)
        indices = self.codes_to_indices(nearest)
        out = self.project_out(z_q)
        return out, indices, nearest
    
    @property
    def codebook_size(self) -> int:
        """Total number of unique quantization points."""
        return int(torch.prod(self.grid_len).item())
    
    @property
    def effective_levels(self) -> List[int]:
        """Grid point counts per pair."""
        return self.grid_len.tolist()


# ─────────────────────────────────────────────
# 4. Q2D2 Projection — MioCodec-compatible wrapper
# ─────────────────────────────────────────────


class Q2D2Projection(nn.Module):
    """
    Q2D2 quantizer + MioCodec-decoder-compatible output projection.
    
    Drop-in replacement for FSQ proj_out in MCS-Trans.
    
    Pipeline:
        encoder_latent  (B, T, enc_dim)
          → latent_head → (B, T, q2d2_dim)       # e.g. 384 → 6
          → Q2D2        → (B, T, q2d2_dim)       # quantized
          → proj_out    → (B, T, content_dim)    # 768 for MioCodec
    """
    
    def __init__(
        self,
        encoder_dim: int = 384,
        q2d2_dim: int = 6,
        content_dim: int = 768,
        levels: List[int] | None = None,
        vq_type: str = "rhombic",
        noise_dropout: float = 0.0,
        gumbel_temperature: float = 0.0,
        use_l2_norm: bool = False,
    ):
        super().__init__()
        
        if levels is None:
            # Default from paper: rhombic with levels=[7,7,7,7,7,7]
            # Matches the 1kbps config; for higher quality use [9,9,9,9,9,9]
            levels = [7, 7, 7, 7, 7, 7]
        
        self.encoder_dim = encoder_dim
        self.q2d2_dim = q2d2_dim
        self.content_dim = content_dim
        
        self.latent_head = nn.Linear(encoder_dim, q2d2_dim)
        self.quantizer = Q2D2Quantizer(
            dim=q2d2_dim,
            levels=levels,
            vq_type=vq_type,
            noise_dropout=noise_dropout,
            gumbel_temperature=gumbel_temperature,
            use_l2_norm=use_l2_norm,
        )
        self.proj_out = nn.Linear(q2d2_dim, content_dim)
        
        codebook_size = self.quantizer.codebook_size
        print(
            f"Q2D2Projection: {encoder_dim}d → {q2d2_dim}d ({self.quantizer.num_pairs} pairs) "
            f"→ {content_dim}d, "
            f"grid={vq_type}, levels={levels}, "
            f"codebook_size={codebook_size:,} "
            f"(compare FSQ 12,800)",
            flush=True,
        )
    
    def forward(
        self, encoder_out: Tensor, return_codes: bool = False
    ) -> Tensor | Tuple[Tensor, Tensor]:
        """
        Args:
            encoder_out: (B, T, encoder_dim) encoder transformer output.
            return_codes: If True, return (content, q2d2_codes).
        
        Returns:
            content: (B, T, content_dim) quantized content for MioCodec decoder.
            q2d2_codes: optional (B, T, q2d2_dim) raw quantizer output.
        """
        latent = self.latent_head(encoder_out)          # (B, T, q2d2_dim)
        quantized, indices = self.quantizer(latent)      # (B, T, q2d2_dim)
        content = self.proj_out(quantized)               # (B, T, content_dim)
        
        if return_codes:
            return content, quantized
        return content
    
    @property
    def codebook_size(self) -> int:
        return self.quantizer.codebook_size


# ─────────────────────────────────────────────
# 5. Diagnostic utilities
# ─────────────────────────────────────────────


def compute_q2d2_perplexity(
    quantizer: Q2D2Quantizer,
    z: Tensor,
) -> dict[str, float]:
    """
    Compute Q2D2 codebook utilization statistics.
    
    Unlike FSQ where each axis is independent, Q2D2 uses
    2D joint distributions, so we report per-pair utilization.
    
    Args:
        quantizer: Q2D2Quantizer instance.
        z: Input tensor (B, T, dim).
    
    Returns:
        Dict with per-pair code usage and overall utilization.
    """
    with torch.no_grad():
        _, _, nearest = quantizer.forward_with_nearest(z)
        # nearest: (B, T, P) — per-pair local grid indices
        
        stats: dict[str, float] = {}
        total_used = 0
        total_available = 0
        
        for i in range(quantizer.num_pairs):
            n_i = nearest[..., i].reshape(-1)
            n_unique = len(torch.unique(n_i))
            n_available = quantizer.grid_len[i].item()
            usage = n_unique / n_available
            
            stats[f"pair_{i}_usage"] = usage
            stats[f"pair_{i}_unique"] = n_unique
            stats[f"pair_{i}_total"] = n_available
            
            total_used += n_unique
            total_available += n_available
        
        stats["overall_usage"] = total_used / total_available
        stats["total_unique"] = total_used
        stats["total_available"] = total_available
        stats["effective_codebook"] = int(
            torch.prod(quantizer.grid_len).item()
        )
        
        return stats


def grid_info(quantizer: Q2D2Quantizer) -> str:
    """Human-readable grid summary."""
    lines = [f"Q2D2 {quantizer.vq_type} grid:"]
    for i in range(quantizer.num_pairs):
        g = quantizer.tile_grid[i]
        lines.append(
            f"  pair {i}: {quantizer.grid_len[i].item()} points, "
            f"range x=[{g[:,0].min():.1f}, {g[:,0].max():.1f}], "
            f"y=[{g[:,1].min():.1f}, {g[:,1].max():.1f}]"
        )
    lines.append(f"  total codebook: {quantizer.codebook_size:,}")
    return "\n".join(lines)
