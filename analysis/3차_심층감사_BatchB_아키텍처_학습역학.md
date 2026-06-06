# 3차 심층 감사 — Batch B: 아키텍처 설계 + 학습 역학

**대상 파일**: `encoder.py` (F³Encoder), `decoder.py` (F³Decoder, MRFBlock, DecoderStage), `train.py` (3-Phase 학습)

**분석 일자**: 2026-06-06  
**분석 범위**: encoder.py 86줄, decoder.py 153줄, train.py 174줄 + 종속 파일 (blocks.py, config.py, converter.py, cfm_loss.py, speaker.py, prosody.py)

---

## 1. Encoder Downsampling 전략 분석

### 현재 설계

| Stage | Channel | Stride | Kernel (stride×3) | 누적 DS | Nyquist 주파수 |
|-------|---------|--------|--------------------|---------|---------------|
| 1 | 32 | 2 | 6 | ×2 | 11,025 Hz |
| 2 | 64 | 2 | 6 | ×4 | 5,512 Hz |
| 3 | 128 | 3 | 9 | ×12 | 1,837 Hz |
| 4 | 256 | 3 | 9 | ×36 | 612 Hz |
| 5 | 512 | 7 | 21 | ×252 | 87 Hz |
| 6 | 768 | 7 | 21 | ×1,764 | 12.5 Hz |

총 다운샘플링: 2×2×3×3×7×7 = **1,764배** → 44,100Hz → **25Hz**

### anti-aliasing 충분성: `kernel_size = stride × 3`

**이론적 기준 (Nyquist-Shannon)**:
- 이상적인 anti-aliasing 필터는 cutoff frequency ≤ f_s / (2 × stride) 를 가져야 함
- `kernel_size ≥ 2 × stride` 는 최소 조건 (커널이 한 주기 이상의 샘플을 커버)
- `kernel_size = stride × 3` 는 이 최소 조건을 1.5배 초과 → **최소 기준은 충족**

**실제 스펙트럼 관점**:
- stride=2, kernel=6: 오버랩 = 4 samples → Nyquist 11kHz. 6-tap 필터로 11kHz 이상을 감쇠시키는 것은 **매우 거친 근사**. 가청 대역(20Hz~20kHz)에서 충분한 감쇠를 보장하지 않음.
- stride=7, kernel=21: 오버랩 = 14 samples → Nyquist ~3.15kHz. 21-tap 필터는 44.1kHz에서 약 0.48ms 길이. 이 정도면 중간 정도의 필터링은 가능하지만 스펙트럼 folding을 완전히 방지할 만큼 sharp한 cutoff를 학습하기는 어려움.

**문제점 (HIGH)**:
```
학습된 strided conv는 anti-aliasing LP filter가 아님 —
손실 함수(L1 reconstruction)에 의해 형성되며 명시적 주파수 제약이 없음.
ConvNeXtV2 블록이 후처리하지만 이미 aliasing된 피처에는 소용 없음.
```

### 소수(prime) factor 7의 스펙트럼 아티팩트 가능성

stride=7이 포함된 총괄 downsampling factor 1764 = 2² × 3² × 7².

**아티팩트 메커니즘**:
- stride=7로 downsampling 시, 입력 신호의 7샘플마다 1샘플을 취하게 됨 (convolutional sampling이므로 weighted average)
- 44.1kHz에서 7의 배수 주파수 (6.3kHz, 12.6kHz, 18.9kHz, ...)가 folding되어 저주파수 대역에 aliasing 유발
- 특히 음성의 fricative, sibilant 에너지가 집중된 4~8kHz 대역이 0~3kHz로 folding될 위험

**Prime stride의 특수성**:
- stride 2, 3은 harmonic series의 일부이므로 배음 구조와 자연스럽게 align될 수 있음
- stride 7은 대부분의 음성 배음 주파수(f₀×n)와 정수배 관계가 아님 → **비조화적(non-harmonic) aliasing 성분** 생성
- 이는 지각적으로 "buzzy"하거나 "metallic"한 아티팩트로 나타날 수 있음

**권고**:
- Stride 7 단계에 **명시적 BlurPool (anti-aliased max pooling)** 또는 low-pass filter pre-convolution 적용
- 또는 stride 7을 (2, 2, 2) 또는 (2, 3)으로 분해 (stage 수 증가)
- 최소한 stride 7 stage 이후에 spectral regularization loss (STFT 기반) 추가 검토

---

## 2. Decoder Upsampling 전략 분석

### TransposedConv + MRFBlock 구조

```
DecoderStage:
  CausalConvTranspose1d(kernel=stride×2+1, stride=stride)
  → MRFBlock(dim, kernels=(3,7,11), dilations=((1,3,5),...))
  → MRFBlock(dim, ...)
  → FiLM(speaker)
```

### Checkerboard 아티팩트 위험성

**ConvTranspose1d의 checkerboard 메커니즘**:
- Transposed convolution에서 `kernel_size % stride ≠ 0` 일 때 출력에 주기적 진폭 변조 발생
- 모든 decoder stage에서 이 조건 위반:

| stride | kernel (stride×2+1) | kernel % stride | Checkerboard 위험 |
|--------|----------------------|-----------------|-------------------|
| 7 | 15 | 1 ≠ 0 | **HIGH** |
| 3 | 7 | 1 ≠ 0 | **HIGH** |
| 2 | 5 | 1 ≠ 0 | **HIGH** |

**구체적 예시 (stride=7, kernel=15)**:
- 입력 feature map의 각 요소가 15개의 출력 위치에 "scatter"됨
- stride=7이므로 scatter 간격은 7
- 15를 7로 나눈 나머지가 1이므로, 인접한 출력 위치들이 받는 contribution 수가 다름
  - 어떤 출력 위치는 2개 input의 overlap 영역, 다른 위치는 3개 input의 overlap 영역
  - 이 불균일이 주기적 패턴 = checkerboard artifact

**MRFBlock의 완화 효과**:
- MRFBlock은 depthwise conv 기반 residual block
- Multi-scale 커널(3, 7, 11)과 dilation(1, 3, 5)으로 다양한 receptive field 커버
- **완화 효과는 제한적** — MRF는 원래 HiFi-GAN에서 multi-scale 패턴 캡처용이지 anti-checkerboard 용도가 아님
- 체커보드 아티팩트는 transposed conv 자체의 구조적 문제이므로 후처리로 완전히 제거하기 어려움

### 큰 stride 먼저 적용하는 이유 분석

Decoder stride 순서: **(7, 7, 3, 3, 2, 2)** — Encoder의 역순.

**해상도 진행**:
```
25Hz  →(×7)→   175Hz  →(×7)→  1,225Hz  →(×3)→  3,675Hz
      →(×3)→ 11,025Hz →(×2)→ 22,050Hz →(×2)→ 44,100Hz
```

**장점** (이론적):
- Autoencoder의 encoder-decoder 대칭성 보존 (mirror architecture)
- 저해상도 latent에서 큰 폭으로 확장 → global structure를 먼저 복원
- 후반부 작은 stride에서 fine detail 복원

**단점** (실제적):
- **Stage 1 (25Hz → 175Hz)**: 가장 압축된 latent에서 7배 업샘플링. 정보가 가장 적은 상태에서 가장 큰 확장을 수행 → **hallucination 위험**
- stride 7 transposed conv의 checkerboard가 가장 낮은 해상도에서 발생 → 이후 모든 stage로 전파됨
- 일반적 관행은 **작은 stride → 큰 stride** 순서 (점진적 확장)

**권고**:
- `kernel_size = stride * 2` (kernel % stride == 0 보장) → checkerboard 원천 제거
- 또는 transposed conv 대신 **subpixel convolution** (nn.PixelShuffle의 1D버전) 또는 **linear interpolation + Conv1d** 조합 사용
- Stride 순서를 (2,2,3,3,7,7)로 변경하여 점진적 확장 검토 (단, latent dim=768 → 16ch로 가는 구조 변경 필요)

---

## 3. CausalConvTranspose1d 출력 트리밍 분석

### 현재 구현

```python
class CausalConvTranspose1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1):
        self.stride = stride
        self.conv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size, stride=stride, padding=0)

    def forward(self, x):
        out = self.conv(x)                    # L_out = (L_in - 1) * S + K
        expected_len = x.shape[2] * self.stride  # L_in * S
        return out[:, :, :expected_len]        # 앞부분만 취함
```

### 기대 출력 길이 분석

ConvTranspose1d(padding=0)의 출력:
```
L_out = (L_in - 1) × stride + kernel_size
```

트리밍 후:
```
L_trimmed = L_in × stride
```

**버려지는 샘플 수**: `L_out - L_trimmed = kernel_size - stride`

| stride | kernel (stride×2+1) | 버려지는 샘플 |
|--------|----------------------|---------------|
| 7 | 15 | 8 |
| 3 | 7 | 4 |
| 2 | 5 | 3 |

총 버려지는 샘플 ≈ 22 (최종 44.1kHz 도메인에서)

### 인과성(Causality) 보장 검증

**Encoder 측 인과적 연산**:
- `CausalConv1d` with padding=(kernel-1, 0): `output[t]`는 `input[t - (kernel-1)*dilation : t*stride]` 범위에만 의존
- Encoder의 마지막 출력 위치 `i`는 입력 시간 `~i*1764`까지의 정보를 포함

**Decoder 측 인과적 연산**:
- `padding=0` ConvTranspose1d: 입력 위치 `i`의 정보가 출력 위치 `[i*S, i*S + K - 1]`로 scatter됨
- 출력 위치 `t` (t < K-1): 입력 위치 0만 참조 → **인과적** (미래 정보 없음)
- 출력 위치 `t` (t ≥ K-1, t < L_in*S): 여러 입력 위치의 overlap → 여전히 인과적
- 출력 위치 `t` (t ≥ L_in*S): 가장 마지막 입력의 right tail만 → 정보 불완전

**트리밍의 효과**:
- `out[:, :, :L_in*S]`: 오른쪽 tail (정보 불완전 구간) 제거 → 인과성에 영향 없음
- 왼쪽 warm-up 구간 (t < K-1)은 유지 → 이 구간은 입력 0의 정보만 사용하므로 인과적이지만, **정보가 불완전**할 수 있음

### `padding=kernel_size-1` 제안 평가

"padding=kernel_size-1을 주고 트리밍을 없애라"는 제안:
```
L_out = (L_in - 1)*S + K - 2*(K-1) = L_in*S - S - K + 2
```

- stride=7, K=15: `L_out = L_in*7 - 7 - 15 + 2 = L_in*7 - 20`
- **필요한 출력 길이(L_in×7)보다 짧아짐** → reconstruction mismatch 발생

**정확한 causal inversion을 위한 조건**:
- Encoder의 `CausalConv1d(kernel=S×3, stride=S)`를 inversion하려면
- Decoder에서 `ConvTranspose1d(kernel=S×3, stride=S, padding=kernel-1)` 사용 후
- `L_out = L_in*S - S + 1` → 여전히 모자람

**결론**:
현재의 트리밍 방식은 **실용적 타협**이다. 완벽한 causal inversion은 encoder의 left-padding으로 인해 불가능하며, 현재 방식은 출력 길이를 L_in × S로 확정하여 downstream과의 호환성을 보장한다.

**경미한 개선안**:
- 트리밍 전에 left side도 kernel_size - stride 만큼 제거하여 출력이 완전한 receptive field를 갖도록 함
- 또는 ConvTranspose1d 대신 **linear interpolation + Conv1d**를 사용하여 checkerboard와 causal alignment를 동시에 해결

---

## 4. 3-Phase 학습 역학 분석

### Phase별 개요

| Phase | 학습 모듈 | 동결 모듈 | Loss | 목적 |
|-------|----------|----------|------|------|
| 0 | Encoder, Decoder | — | L1(recon, src) | AE 사전학습 |
| 1 | VFN (VectorFieldNet) | Encoder, Decoder, SpeakerEnc, Prosody | CFM MSE | Flow 학습 |
| 2 | Encoder, Decoder, VFN | SpeakerEnc, Prosody | L1(out, tgt) | E2E 미세조정 |

### Phase 0: AE 재구성의 문제점 (CRITICAL)

**Phase 0의 loss 구성**:
```python
z = encoder(src, training=True)          # src waveform → latent
spk_emb, prompt = speaker_enc(ref)       # REFERENCE speaker embedding
recon = decoder(z, spk_emb)              # latent + ref speaker → waveform
loss = F.l1_loss(recon, src)             # reconstruct SOURCE waveform!
```

**근본적 모순**:
- `z`는 **src 화자**의 음성에서 추출된 latent
- `spk_emb`는 **ref 화자**의 speaker embedding (VCTK 데이터셋에서 ref는 타겟 화자)
- Decoder는 "src의 내용 + ref의 화자" 정보로 **src 파형을 복원**해야 함
- 이는 decoder가 **화자 정보를 무시하고 latent에 의존**하도록 학습시킴

**결과적 영향**:
1. **Disentanglement 실패**: Latent z가 화자 정보를 포함해야만 decoder가 src를 복원 가능 → z는 content + speaker 혼합 표현
2. **Decoder의 FiLM 조건화 무력화**: Decoder는 학습 과정에서 "FiLM이 주는 화자 정보 ≠ 출력 화자"를 경험 → FiLM을 신뢰하지 않게 됨
3. **Phase 1 악영향**: VFN이 학습하는 latent 공간이 content-only가 아니므로, flow 변환이 content 보존 + speaker 변환을 동시에 수행해야 함 (난이도 상승)

**Phase 0 개선 방안**:
```python
# 제안 1: src = ref (자기 재구성)
z = encoder(src, training=True)
spk_emb, _ = speaker_enc(src)          # 동일 화자!
recon = decoder(z, spk_emb)
loss = F.l1_loss(recon, src)

# 제안 2: ref 기반 reconstruction + speaker consistency loss 추가
z = encoder(src, training=True)
spk_emb_src, _ = speaker_enc(src)
spk_emb_ref, _ = speaker_enc(ref)
recon_self = decoder(z, spk_emb_src)
loss_recon = F.l1_loss(recon_self, src)
# + speaker consistency: decoder(z, spk_emb_ref)와 src의 content 유사도
```

### Phase 1: CFM 학습의 latent 적합성

**CFM의 동작**:
- `z_src = encoder.encode(src)`, `z_tgt = encoder.encode(tgt)` (noise 없음)
- `z_t = (1-t)*z_src + t*z_tgt + σ_min·ε` (OT path)
- `v_target = z_tgt - z_src` (constant velocity field)
- VFN이 `v_θ(z_t, t, c)` → `z_tgt - z_src`를 예측하도록 학습

**Latent 공간 품질 요구사항**:
- 이상적: `z_src`와 `z_tgt`가 content는 동일하고 speaker만 다른 표현
- 현실 (Phase 0 설계 결함으로): `z_src`와 `z_tgt`에 content와 speaker 정보가 혼재
- CFM은 직선 보간(OT path)을 가정하는데, 혼합 표현의 직선 보간이 semantically valid한지 의문

**직선 보간의 문제**:
- Content+speaker 혼합 공간에서 `(1-t)*z_src + t*z_tgt`는 t=0.5일 때 두 화자의 중간 목소리 + 중간 내용이 됨
- 이 "중간 상태"가 실제 voice conversion의 물리적 과정을 대표하는가? → **의문**
- Disentangled 표현이었다면: content는 불변, speaker만 interpolate → 훨씬 자연스러운 flow path

### Phase 2: E2E 미세조정의 gradient path

```python
z_src = encoder.encode(src)           # grad flows (no noise)
z_tgt = solve_cfm_euler(vfn, ...)     # grad flows through 4 Euler steps
out = decoder(z_tgt, spk_emb)         # grad flows
loss = F.l1_loss(out, tgt)
```

**Gradient 흐름**: loss → decoder → solver(4 VFN calls) → encoder

**문제점**:
1. **ODE solver 통한 역전파**: 4-step Euler + half-step refinement → 총 5회의 VFN forward → 메모리 사용량 5배, gradient instability 위험
2. **Encoder의 이중 역할**: Phase 2에서 encoder는 "content 추출"과 "flow-friendly latent 생성"을 동시에 학습 → 목표 충돌 가능성
3. **Noise regularization 부재**: `encoder.encode()`는 noise 없음 → Phase 0에서 학습한 noise robustness가 Phase 2에서 활용되지 않음

---

## 5. Optimizer 분석: betas=(0.8, 0.9), weight_decay=0.01

### betas 비교

| 파라미터 | FlowVC 값 | PyTorch 기본값 | 일반적 관행 |
|----------|----------|---------------|------------|
| β₁ | **0.8** | 0.9 | 0.9 |
| β₂ | **0.9** | 0.999 | 0.999 |

**β₁ = 0.8 (첫 번째 모멘텀 감쇠율)**:
- 낮은 β₁ → 최근 그래디언트에 더 빠르게 적응, 과거 모멘텀 빠르게 소멸
- **장점**: batch_size=1의 고분산 그래디언트에서 과거 노이즈가 현재 step을 오염시키는 것 방지
- **단점**: 모멘텀 효과 감소 → saddle point 탈출 느려짐, long-term consistency 저하
- batch_size=1 환경에서는 **합리적 선택**

**β₂ = 0.9 (두 번째 모멘텀 감쇠율)**:
- PyTorch 기본값 0.999 대비 **급격한 적응 속도**
- Adam의 adaptive learning rate: `lr / (√v̂ + ε)` — β₂가 낮을수록 v̂(제곱 그래디언트 평균)이 빠르게 변함
- **장점**: 3-phase 학습에서 phase 전환 시 loss landscape 변화에 빠르게 적응
- **단점**: 
  - 학습 후반부에 learning rate 변동성 증가 → 수렴 불안정
  - Sparse gradient 환경에서 분모가 너무 빨리 변해 step size가 erratic
  - 94M 파라미터 규모에서는 일부 파라미터의 그래디언트가 드물게 발생 → instability 위험

**권고**: β₂ = 0.95 또는 0.98로 상향 조정 검토. Phase 전환 시에만 optimizer state reset을 고려.

### weight_decay = 0.01 분석

- AdamW에서 weight_decay는 L2 정규화와 유사하지만 gradient와 decoupled
- 94M 파라미터, batch_size=1, steps=200K 상황에서:
  - weight_decay=0.01 → 매 step마다 weight가 0.01 × lr 만큼 감쇠
  - lr=2e-4 → 유효 감쇠율 = 2e-6/step → 200K step 후 약 33% 감쇠
- **적절한 범위**: 일반적으로 0.0001 ~ 0.1. 0.01은 중간 정도.
- batch_size=1 + 상대적으로 높은 weight_decay → **과도한 정규화 위험**
- 특히 encoder-decoder가 Phase 0에서만 200K step 학습할 때, over-regularization으로 표현력 제한 가능

**권고**: Phase 0에서는 weight_decay=0.01, Phase 1에서는 0.001, Phase 2에서는 0.005로 phase별 차등 적용 검토.

---

## 6. Phase 1 Freeze 전략 분석

### LayerNorm Running Stats 동결

```python
encoder.eval()           # → LayerNorm이 running stats 사용, 업데이트 중단
p.requires_grad = False  # → gamma, beta 파라미터 동결
```

**Running stats 업데이트 메커니즘** (PyTorch 내부):
- `training=True`일 때만 `running_mean`, `running_var`가 momentum 업데이트됨
- `eval()` 호출 시 running stats는 고정, batch statistics 대신 사용
- `requires_grad=False`는 파라미터(gamma, beta)만 동결, running stats와 무관

**Phase 0 수렴 가정의 타당성**:

| 가정 | 평가 |
|------|------|
| Encoder가 충분히 수렴했음 | Phase 0이 200K steps이면 **대체로 타당** |
| Running stats가 안정적임 | **조건부 타당** — batch_size=1로 인한 분산이 stats에 누적되었을 수 있음 |
| Phase 1 데이터 분포가 Phase 0과 동일 | **비타당** — Phase 0은 src=src 재구성(?), Phase 1은 src→tgt pair 사용 |

**주의점**:
- Encoder의 `ConvNeXtV2Block` 내부에 LayerNorm이 있음 (DWConv 후 channel-wise LN)
- 이 LayerNorm의 running stats가 Phase 0의 입력 분포(src 음성)에 맞춰져 있음
- Phase 1에서는 src와 tgt 모두 encode → tgt의 분포가 미묘하게 다를 수 있음
- 그러나 encoder를 통한 latent 추출은 결정론적(deterministic)이므로, running stats의 경미한 mismatch는 큰 문제가 되지 않음

**Phase 2에서의 LayerNorm 상태**:
```python
# Phase 2: encoder는 training mode (eval() 호출 안 함)
z_src = encoder.encode(src)  # encode() → forward(training=False)
```
- `encoder.encode()`는 `forward(training=False)` 호출 → LayerNorm은 running stats 사용
- 하지만 `encoder` 자체는 `training=True` 상태 → running stats는 **업데이트되고 있음**!
- 이는 **불일치**: forward에서는 running stats를 사용하지만, backward에서는 stats가 업데이트됨
- PyTorch의 `training` flag와 `forward(training=...)` 인자의 상호작용에 주의 필요

**실제 동작 추적**:
- `encoder.encode(wav)` → `self.forward(wav, training=False)` → `self.stages(wav)` 호출
- `self.stages`는 `nn.Sequential` → 각 ConvNeXtV2Block.forward() 호출
- LayerNorm은 `self.training` (모듈 상태) 기준으로 batch stats vs running stats 결정
- `encoder.encode()` 내부에 `self.eval()` 또는 context manager가 없으므로
- **encoder.training이 True면 LayerNorm은 batch stats 사용 + running stats 업데이트**
- training=False 인자는 noise regularization에만 영향 (forward() 함수의 인자)
- → **실제로는 training=True 상태로 running stats가 업데이트됨**

이것은 의도된 동작일 수도 있지만, `self.training` 플래그와 `training` 함수 인자의 의미가 분리되어 있어 혼란을 야기할 수 있다.

---

## 종합 평가

### 설계 강점
1. **완전 인과적 구조**: CausalConv1d, CausalConvTranspose1d로 실시간 추론 가능
2. **ConvNeXt v2**: GRN + LayerScale + zero-init으로 안정적 심층 학습
3. **FiLM + zero-init**: 학습 초기에 speaker conditioning이 identity로 시작 → 안정적 수렴
4. **CFM with OT path**: 직선 경로로 단순하고 효율적인 flow matching

### 설계 약점 (심각도 순)

| # | 문제 | 심각도 | 영향 |
|---|------|--------|------|
| 1 | Phase 0의 src≠ref 재구성 | **CRITICAL** | Latent disentanglement 원천적 실패 |
| 2 | ConvTranspose1d checkerboard (모든 stage) | **HIGH** | 출력 음질 저하, 주기적 아티팩트 |
| 3 | Stride-7 aliasing without anti-aliasing | **HIGH** | 고주파수 folding → 금속성 아티팩트 |
| 4 | 큰 stride 우선 업샘플링 | **MEDIUM** | 저해상도 hallucination 전파 |
| 5 | β₂=0.9의 과도한 적응성 | **MEDIUM** | 수렴 안정성 저하 가능 |
| 6 | Phase 2의 noise regularization 누락 | **LOW** | Robustness 저하, train-inference gap |
| 7 | CausalConvTranspose1d 왼쪽 warm-up | **LOW** | 초기 프레임 품질 저하 |

### 우선 조치 권고

1. **Phase 0 수정** (즉시): `ref → src`로 speaker embedding 변경 → 자기 재구성 학습
2. **ConvTranspose1d 개선**: kernel_size를 stride의 정수배로 변경 또는 interpolation + Conv1d로 대체
3. **Anti-aliasing 추가**: stride≥3 stage에 low-pass filter 또는 BlurPool 도입
4. **β₂ 조정**: 0.95로 변경하고 Phase 전환 시 optimizer state 재초기화 검토
