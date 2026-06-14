# FlowVC 코드 감사 — 종합 보고서

> Legacy note: this report audits the removed FlowVC/F³ implementation, not the
> current `astrape/` MioCodec student code.

> 2026-06-06 | 3개 병렬 에이전트 | 10개 파일 전수 감사

---

## 🔴 CRITICAL — 당장 고쳐야 함 (13건)

| # | 파일:라인 | 이슈 | 영향 |
|---|----------|------|------|
| 1 | `blocks.py:84` | **GRN 수식 완전 오류** — `dim=(1,2)`로 채널+시간 동시 collapse → 사실상 no-op. ConvNeXt v2 핵심 정규화 무력화 | 인코더/디코더/VFN 전체 |
| 2 | `converter.py:276` | **out_gate=0 → 전체 네트워크 기울기 차단**. VFN 37M이 학습 불가 상태 | VFN 전체 |
| 3 | `train.py:65-177` | **Phase 0/1/2 전부 smoke test only**. `torch.randn` 더미데이터 1회 forward. 실제 학습 루프 없음 | 학습 불가 |
| 4 | `prosody.py:22-79` | **ProsodyExtractor가 가짜 구현**. 4-layer ConvNet일 뿐, 실제 F0 추출 알고리즘 없음. supervised 신호도 없어 아무 의미 없는 특징 출력 | 운율 전체 |
| 5 | `infer.py:97-139` | **스트리밍 버퍼 미사용**. 모든 청크 독립처리 → 청크 경계에서 causal conv 수용영역 부족 → click/glitch | 추론 품질 |
| 6 | `train.py:204-217` | **체크포인트 저장/로드 로직 전무**. `--resume` 플래그 있어도 새 모델 생성. 학습 결과 전부 버려짐 | 학습 불가 |
| 7 | `dataset.py:218-240` | **build_cache가 비결정적**. `random.choice`로 페어링 → 동일 인덱스 재로드 시 다른 데이터. 캐시 무용지물 | 데이터 |
| 8 | `blocks.py:61-65` | **CausalConvTranspose1d 길이 트리밍 부정확**. `kernel_size - stride` 샘플씩 누락, 6단계 누적 시 수백 샘플 오차 | 디코더 |
| 9 | `encoder.py:57` | **training 기본값 True**. `model(wav)` 호출 시 항상 노이즈 추가. `model.eval()` 무시됨 | 추론 품질 |
| 10 | `infer.py:141-170` | **청크 간 불연속 + zero-padding 왜곡**. non-overlapping 처리로 경계 아티팩트 | 추론 품질 |
| 11 | `dataset.py:105` | **오디오 repeat 패딩**. 짧은 파일 반복 시 위상 불연속(phase discontinuity) → 틱 노이즈 학습 | 데이터 |
| 12 | `dataset.py:130` | **참조 오디오 데이터 누수**. src_path가 ref로 선택될 수 있음 → 화자 변환 불가 | 데이터 |
| 13 | `infer.py:362` | **파일 변환 시 무작위 초기화 모델 사용**. `--checkpoint` 플래그 없음. 학습된 가중치 로딩 불가 | 추론 불가 |

---

## 🟠 HIGH — 심각 (10건)

| # | 파일 | 이슈 |
|---|------|------|
| 14 | `converter.py:134` | FlowBlock LayerScale gamma=0 → 12블록 순차적 해동, 수렴 지연 |
| 15 | `converter.py:89` | SpeakerCrossAttn proj_kv 비대칭 초기화 (K/V 랜덤, O 제로) |
| 16 | `cfm_loss.py:17` | ODE solver import 구조 불량 (loss가 solver에 의존) |
| 17 | `decoder.py:96` | speaker_dim=192 하드코딩. config 변경 시 불일치 |
| 18 | `infer.py:288-326` | 프로파일러 GPU sync 누락 + 웜업 3회만 (10-50회 필요) |
| 19 | `converter.py:286` | Euler solver t=1 평가 누락 (마지막 스텝 t=0.75) |
| 20 | `train.py:147-149` | Phase 2 optimizer에서 speaker_enc/prosody 제외 — docstring과 불일치 |
| 21 | `cfm_loss.py:128` | FlowVCLoss가 ODE solver 호출 → n_steps 하드코딩(4) |
| 22 | `decoder.py` ↔ `blocks.py` | CausalConvTranspose1d 트리밍과 MRFBlock causal 일관성 미검증 |
| 23 | `config.py` | TrainConfig 있으나 train.py에서 전혀 사용 안 함 |

---

## 🟡 MEDIUM (19건) / 🟢 LOW (18건)

주요 항목:
- `converter.py`: dropout 미사용, n_steps=0 divide-by-zero
- `decoder.py`: MRF 평균화 vs 합산, DecoderStage docstring 불일치  
- `cfm_loss.py`: sigma_min 고정, float/Tensor 혼합
- `speaker.py`: 어텐션 풀링 정규화 누락
- `dataset.py`: 데드코드(self.sr_orig), 하드코딩 개인 경로
- `infer.py`: torchaudio.resample GPU crash 가능성
- 전체: 일부 일본어 주석 혼재, type hint 누락

---

## 📊 통계

| 심각도 | 개수 |
|--------|:---:|
| CRITICAL | 13 |
| HIGH | 10 |
| MEDIUM | 19 |
| LOW | 18 |
| **총합** | **60** |

---

## 🎯 수정 우선순위 Top 5

1. **GRN 수식 수정** (`blocks.py:84`): `dim=(1,2)` → `dim=-1` 한 글자
2. **out_gate 초기화** (`converter.py:276`): `torch.zeros(1)` → `torch.ones(1)*0.01`
3. **train.py 실제 학습 루프 구현**: 데이터로더 연동 + checkpoint save/load
4. **prosody.py 실제 F0 추출기로 교체**: btrv3lite `f0.py`의 RMVPE/PENN 연동
5. **infer.py 스트리밍 버퍼 + overlap-add 구현**: 청크 경계 아티팩트 제거

---

*감사 실행: 2026-06-06, 3 parallel agents*
